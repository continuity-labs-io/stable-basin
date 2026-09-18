from jaxtyping import PRNGKeyArray
from jaxtyping import jaxtyped
from beartype import beartype
import jax
import jax.numpy as jnp
import equinox as eqx
import torx
import torx.factor
import logging

from src.echo.architecture.markov_hull import MarkovHull
from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.physics.solenoidal import SolenoidalFlow
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.physics.thermostat import Thermostat
from src.echo.primitives.thermalizer import TorxThermalizer, ForcedTorxThermalizer

class MaskedThermoFlowFactor(torx.factor.AbstractReferenceFactor):
    """
    Custom Torx factor that applies the Markov Blanket topological mask to the
    underlying physical matrices to enforce conditional independence.
    
    A Note on Factor Graphs: In traditional computer science graph theory, a
    "Factor Graph" is a bipartite graph with two types of nodes: "Variable
    Nodes" (representing state/data) and "Factor Nodes" (representing
    computations or constraints applied to that data).
    
    By inheriting from `AbstractReferenceFactor`, this class defines a single
    computational "Factor Node". In our simulation, it takes in the current
    state variables (`x` and `dt`), computes the physical thermodynamic step,
    and outputs the resulting next state variable.
    """
    ebm: PrecisionWeightedEBM
    solenoidal: SolenoidalFlow
    dissipative: DissipativeFriction
    thermostat: Thermostat
    hull: MarkovHull
    d_state: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    use_blanket_topology: bool = eqx.field(static=True)
    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self, ebm, solenoidal, dissipative, thermostat, hull, d_state, epsilon, use_blanket_topology):
        self.ebm = ebm
        self.solenoidal = solenoidal
        self.dissipative = dissipative
        self.thermostat = thermostat
        self.hull = hull
        self.d_state = d_state
        self.epsilon = epsilon
        self.use_blanket_topology = use_blanket_topology
        
        self.input_ports = {
            "x": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "omega_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "q_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((d_state,), jnp.float32)

    def init_params(self, key):
        # Required by torx.factor.AbstractReferenceFactor interface
        return {}
        
    def precompute(self) -> dict:
        """
        Precomputes and hoists O(D^3) matrix constructions out of the ODE loop.
        """
        Gamma = self.dissipative.Gamma
        return {
            "Q": self.solenoidal.Q,
            "L": jnp.linalg.cholesky(Gamma),
            "Gamma": Gamma
        }

    @jaxtyped(typechecker=beartype)
    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        """
        Executes a single discrete integration step of the physical thermodynamic factor.
        
        Args:
            key: JAX PRNG key for stochastic sampling (e.g., Langevin noise).
            inputs: Dictionary containing the necessary state variables:
                - "x": The current state vector.
                - "dt": The time delta for the integration step.
                - "omega_ext": (Optional) External solenoidal forcing.
                - "q_ext": (Optional) External heat injection.
            params: Parameters dictionary (unused, required by torx interface).
            info: Optional factor info (unused, required by torx interface).
            site_info: Optional site info (unused, required by torx interface).
            return_aux: If True, returns a tuple of (next_state, auxiliary_data).
                Required by the torx AbstractReferenceFactor signature.
                
        Returns:
            x_next: The integrated state vector for the next time step. If 
                return_aux is True, returns (x_next, None).
        """
        x = inputs["x"]
        dt = inputs["dt"]
        omega_ext = inputs.get("omega_ext", jnp.zeros(self.d_state, dtype=jnp.float32))
        q_ext = inputs.get("q_ext", jnp.zeros(self.d_state, dtype=jnp.float32))

        def energy_fn(state):
            state_obs = self.hull.apply_sensory_degradation(state)
            e, _ = self.ebm(state_obs)
            return e

        grad_E = jax.grad(energy_fn)(x)
        
        params = params or {}
        Q = params.get("Q", self.solenoidal.Q)
        Gamma = params.get("Gamma", self.dissipative.Gamma)
        L = params.get("L", jnp.linalg.cholesky(Gamma))
        
        x_next = self.thermostat(
            x=x,
            grad_E=grad_E,
            Q=Q,
            L=L,
            dt=dt,
            key=key,
            omega_ext=omega_ext,
            q_ext=q_ext,
            Gamma=Gamma
        )
        
        if return_aux:
            return x_next, None

        return x_next


class MarkovBlanketObserver(eqx.Module):
    """
    Fuses physical boundaries, energy-based learning, and stochastic unrolling
    into a single localized self-evidencing entity.
    """
    hull: MarkovHull
    ebm: PrecisionWeightedEBM
    solenoidal: SolenoidalFlow
    dissipative: DissipativeFriction
    thermostat: Thermostat
    thermalizer: TorxThermalizer
    forced_thermalizer: ForcedTorxThermalizer
    use_blanket_topology: bool = eqx.field(static=True)

    def __init__(
        self,
        d_internal: int,
        d_sensory: int,
        d_active: int,
        d_external: int,
        ebm_hidden_size: int,
        ebm_depth: int,
        n_steps: int,
        temperature: float,
        key: PRNGKeyArray,
        D_s: jax.Array | None = None,
        epsilon: float = 1e-4,
        use_blanket_topology: bool = True
    ):
        self.hull = MarkovHull(d_internal, d_sensory, d_active, d_external, D_s=D_s)
        d_state = self.hull.d_state
        self.use_blanket_topology = use_blanket_topology
        
        if use_blanket_topology:
            logging.info("Initializing MarkovBlanketObserver with strictly positive-definite partitioned topology.")
        else:
            logging.info("Initializing MarkovBlanketObserver with unpartitioned full-rank dense topology.")
        
        k1, k2, k3 = jax.random.split(key, 3)
        self.ebm = PrecisionWeightedEBM(
            d_state=d_state,
            hidden_size=ebm_hidden_size,
            depth=ebm_depth,
            key=k1
        )
        self.solenoidal = SolenoidalFlow(d_state=d_state, key=k2, hull=self.hull if use_blanket_topology else None)
        self.dissipative = DissipativeFriction(d_state=d_state, key=k3, hull=self.hull if use_blanket_topology else None)
        self.thermostat = Thermostat(temperature=temperature)
        
        masked_factor = MaskedThermoFlowFactor(
            ebm=self.ebm,
            solenoidal=self.solenoidal,
            dissipative=self.dissipative,
            thermostat=self.thermostat,
            hull=self.hull,
            d_state=d_state,
            epsilon=epsilon,
            use_blanket_topology=use_blanket_topology
        )
        
        self.thermalizer = TorxThermalizer(
            flow_factor=masked_factor, 
            n_steps=n_steps, 
            d_state=d_state
        )
        self.forced_thermalizer = ForcedTorxThermalizer(
            flow_factor=masked_factor,
            d_state=d_state,
            injection_start_idx=self.hull.d_internal
        )

    @jaxtyped(typechecker=beartype)
    def __call__(self, key: PRNGKeyArray, x_init: jax.Array, dt: float) -> jax.Array:
        """
        Executes the unrolled simulation over n_steps.
        """
        factor_params = self.thermalizer.graph.sites[0].factor.base.precompute()
        return self.thermalizer(key, x_init, dt, factor_params=factor_params)

    def forced_unroll(self, key: PRNGKeyArray, x_init: jax.Array, dt: float, seq: jax.Array | None = None, omega_seq: jax.Array | None = None, q_gain: float = 0.0, q_mask: jax.Array | None = None) -> jax.Array:
        """
        Executes the unrolled simulation over an external sequence.
        """
        factor_params = self.forced_thermalizer.flow_factor.precompute()
        return self.forced_thermalizer(key, x_init, dt, seq=seq, omega_seq=omega_seq, q_gain=q_gain, q_mask=q_mask, factor_params=factor_params)

    def extract_internal_state(self, x: jax.Array) -> dict:
        """
        Extracts the internal partitions of a given state vector.
        """
        return self.hull.partition(x)
