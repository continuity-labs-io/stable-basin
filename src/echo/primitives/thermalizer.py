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
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "omega_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "q_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)
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
        omega_ext = inputs.get("omega_ext", jnp.zeros(self.d_state, dtype=jnp.float32))
        q_ext = inputs.get("q_ext", jnp.zeros(self.d_state, dtype=jnp.float32))

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
            key=key,
            omega_ext=omega_ext,
            q_ext=q_ext
        )
        
        if return_aux:
            return x_next, None
        return x_next

class TorxThermalizer(eqx.Module):
    """
    A temporal unroller (or numerical integration loop) for the simulation.
    
    While "Thermalizer" is a term from physics (referring to a system reaching 
    thermal equilibrium over time), in standard computational terms, this class acts 
    as a Recurrent Neural Network (RNN) unroller or a numerical ODE solver loop.
    
    It takes a single-step physical transition function (the factor) and applies it 
    iteratively over a fixed number of continuous timesteps (`n_steps`), chaining 
    the outputs together to produce a full time-series trajectory.
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
                    parents=("x_init", "dt_constant", "omega_ext_constant", "q_ext_constant"),
                    porting_fn=("x", "dt", "omega_ext", "q_ext"),
                    param_key=None,
                    info_key=None,
                    site_info=None
                ),
            ),
            input_ports={
                "x_init": jax.ShapeDtypeStruct((d_state,), jnp.float32),
                "dt_constant": jax.ShapeDtypeStruct((), jnp.float32),
                "omega_ext_constant": jax.ShapeDtypeStruct((d_state,), jnp.float32),
                "q_ext_constant": jax.ShapeDtypeStruct((d_state,), jnp.float32)
            },
            output_name="chain"
        )
        
    @eqx.filter_jit
    def __call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float) -> jax.Array:
        """
        Executes the unrolled simulation.
        """
        inputs = {
            "x_init": x_init,
            "dt_constant": jnp.array(dt, dtype=jnp.float32),
            "omega_ext_constant": jnp.zeros_like(x_init),
            "q_ext_constant": jnp.zeros_like(x_init)
        }
        return self.graph.sample(key, inputs=inputs, params={})

class ForcedTorxThermalizer(eqx.Module):
    """
    A temporal unroller for forced (open) systems driven by an external time-series sequence.
    """
    flow_factor: torx.factor.AbstractReferenceFactor
    d_state: int = eqx.field(static=True)
    injection_start_idx: int = eqx.field(static=True)

    def __init__(self, flow_factor: torx.factor.AbstractReferenceFactor, d_state: int, injection_start_idx: int):
        self.flow_factor = flow_factor
        self.d_state = d_state
        self.injection_start_idx = injection_start_idx

    @eqx.filter_jit
    def __call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float, seq: jax.Array | None = None, omega_seq: jax.Array | None = None, q_gain: float = 0.0, q_mask: jax.Array | None = None) -> jax.Array:
        """
        Executes the unrolled simulation with external forcing and closed-loop control.
        """
        seq_len = None
        for s in (seq, omega_seq):
            if s is not None:
                seq_len = s.shape[0]
                break
                
        if seq_len is None:
            raise ValueError("Must provide either seq or omega_seq.")
            
        def step_fn(state, carry):
            data_frame, omega_frame, step_key = carry
            
            if seq is not None:
                state = jax.lax.dynamic_update_slice(state, data_frame, (self.injection_start_idx,))
                
            current_q_mask = q_mask if q_mask is not None else jnp.ones_like(state)
            
            # Closed-Loop Proportional Controller
            # This calculates a dynamic restorative force that pulls the state towards the origin (homeostasis).
            # It is a closed-loop system because the force (`q_ext`) adapts at each timestep based on the
            # current `state`. The `current_q_mask` ensures that this external actuation is only applied
            # to accessible physical components (e.g., the Markov Blanket: sensory and active states),
            # relying on the network's internal physics to drag the unactuated internal states to safety.
            q_ext = -q_gain * state * current_q_mask
            
            inputs = {
                "x": state,
                "dt": jnp.array(dt, dtype=jnp.float32),
                "omega_ext": omega_frame if omega_seq is not None else jnp.zeros_like(x_init),
                "q_ext": q_ext
            }
            next_state = self.flow_factor.sample(step_key, inputs=inputs, params={})
            return next_state, next_state

        keys = jax.random.split(key, seq_len)
        dummy_seq = jnp.zeros((seq_len, 1))
        
        scan_seq = seq if seq is not None else dummy_seq
        scan_omega = omega_seq if omega_seq is not None else dummy_seq
        
        _, traj = jax.lax.scan(step_fn, x_init, (scan_seq, scan_omega, keys))
        return traj
