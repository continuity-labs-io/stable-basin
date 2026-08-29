import jax
import jax.numpy as jnp
import equinox as eqx
import torx
import torx.factor

from src.echo.physics.solenoidal import SolenoidalFlow
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.physics.thermostat import Thermostat
from src.echo.primitives.ebm import PrecisionWeightedEBM

class ThermoFlowFactor(torx.factor.AbstractReferenceFactor):
    """
    Torx factor representing a single, stochastic continuous-time update step
    of our biological simulation using the Euler-Maruyama method.
    """
    
    ebm: PrecisionWeightedEBM
    solenoidal: SolenoidalFlow
    dissipative: DissipativeFriction
    thermostat: Thermostat
    d_state: int

    # Torx port specifications
    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(
        self,
        ebm: PrecisionWeightedEBM,
        solenoidal: SolenoidalFlow,
        dissipative: DissipativeFriction,
        thermostat: Thermostat,
        d_state: int
    ):
        """
        Initializes the ThermoFlowFactor with required physics and EBM components.
        
        Args:
            ebm: PrecisionWeightedEBM instance.
            solenoidal: SolenoidalFlow instance.
            dissipative: DissipativeFriction instance.
            thermostat: Thermostat instance.
            d_state: Dimensionality of the state vector.
        """
        self.ebm = ebm
        self.solenoidal = solenoidal
        self.dissipative = dissipative
        self.thermostat = thermostat
        self.d_state = d_state
        
        # Define static input and output specs for Torx factor
        self.input_ports = {
            "x": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((d_state,), jnp.float32)

    def init_params(self, key: jax.random.PRNGKey) -> dict:
        """
        Initializes the Torx factor parameters. Since state is tracked via 
        Equinox modules, this returns an empty dictionary.
        """
        return {}

    def sample(
        self,
        key: jax.random.PRNGKey,
        inputs: dict,
        params: dict,
        info: dict = None,
        site_info: dict = None,
        return_aux: bool = False
    ):
        """
        Evaluates the SDE using Euler-Maruyama via the underlying physics modules.
        
        Args:
            key: PRNGKey for generating stochastic noise.
            inputs: Dictionary containing 'x' and 'dt'.
            params: Dictionary of parameters (unused as Equinox handles state).
            info: Optional auxiliary info.
            site_info: Optional site info for the probabilistic trace.
            return_aux: Whether to return auxiliary data (x_next, None).
            
        Returns:
            The next state vector x_next, or a tuple (x_next, None) if return_aux is True.
        """
        x = inputs["x"]
        dt = inputs["dt"]

        # 1. Compute the local slope of the energy landscape
        def energy_fn(state):
            # Evaluate EBM and return only the scalar energy
            e, _ = self.ebm(state)
            return e
            
        grad_E = jax.grad(energy_fn)(x)
        
        # 2. Extract the physical matrices
        Q = self.solenoidal.Q
        L = jnp.tril(self.dissipative.W)
        
        # 3. Execute the physical integration (Thermostat)
        x_next = self.thermostat(
            x=x,
            grad_E=grad_E,
            Q=Q,
            L=L,
            dt=dt,
            key=key
        )
        
        if return_aux:
            return x_next, None
        return x_next

class TorxThermalizer(eqx.Module):
    """
    Continuous-time compiler for the biological simulation,
    unrolling the ThermoFlowFactor sequentially over n_steps.
    """
    n_steps: int = eqx.field(static=True)
    graph: torx.DFG

    def __init__(self, flow_factor: ThermoFlowFactor, n_steps: int, d_state: int):
        self.n_steps = n_steps
        
        # Create ChainFactor unrolling over n_steps
        chain_factor = torx.ChainFactor(
            base=flow_factor,
            n_steps=n_steps,
            feedback_porting_fn="x",
            weight_tied=True
        )
        
        # Wrap the ChainFactor in a Torx DFG
        self.graph = torx.DFG(
            sites=(
                torx.Site(
                    name="chain",
                    factor=chain_factor,
                    parents=("x_init", "dt_constant"),
                    porting_fn=("x", "dt"),
                    param_key=None,
                    info_key=None,
                    site_info=None
                ),
            ),
            input_ports={
                "x_init": jax.ShapeDtypeStruct((d_state,), jnp.float32),
                "dt_constant": jax.ShapeDtypeStruct((), jnp.float32)
            },
            output_name="chain"
        )
        
    @eqx.filter_jit
    def __call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float) -> jax.Array:
        """
        Executes the unrolled simulation.
        """
        inputs = {"x_init": x_init, "dt_constant": jnp.array(dt, dtype=jnp.float32)}
        return self.graph.sample(key, inputs=inputs, params={})
