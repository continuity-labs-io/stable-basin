import jax
import jax.numpy as jnp
import equinox as eqx
import torx
import torx.factor

from src.echo.architecture.markov_hull import MarkovHull
from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.physics.solenoidal import SolenoidalFlow
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.physics.thermostat import Thermostat
from src.echo.primitives.thermalizer import TorxThermalizer, ForcedTorxThermalizer

class MaskedThermoFlowFactor(torx.factor.AbstractReferenceFactor):
    """
    Custom Torx factor that applies the Markov Blanket topological mask
    to the underlying physical matrices to enforce conditional independence.
    
    A Note on Factor Graphs:
    In traditional computer science graph theory, a "Factor Graph" is a bipartite 
    graph with two types of nodes: "Variable Nodes" (representing state/data) and 
    "Factor Nodes" (representing computations or constraints applied to that data).
    
    By inheriting from `AbstractReferenceFactor`, this class defines a single 
    computational "Factor Node". In our simulation, it takes in the current state 
    variables (`x` and `dt`), computes the physical thermodynamic step, and outputs 
    the resulting next state variable.
    """
    ebm: PrecisionWeightedEBM
    solenoidal: SolenoidalFlow
    dissipative: DissipativeFriction
    thermostat: Thermostat
    hull: MarkovHull
    d_state: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self, ebm, solenoidal, dissipative, thermostat, hull, d_state, epsilon):
        self.ebm = ebm
        self.solenoidal = solenoidal
        self.dissipative = dissipative
        self.thermostat = thermostat
        self.hull = hull
        self.d_state = d_state
        self.epsilon = epsilon
        
        self.input_ports = {
            "x": jax.ShapeDtypeStruct((d_state,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "omega_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((d_state,), jnp.float32)

    def init_params(self, key):
        return {}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        x = inputs["x"]
        dt = inputs["dt"]
        omega_ext = inputs.get("omega_ext", jnp.zeros(self.d_state, dtype=jnp.float32))

        def energy_fn(state):
            state_obs = self.hull.apply_sensory_degradation(state)
            e, _ = self.ebm(state_obs)
            return e
            
        grad_E = jax.grad(energy_fn)(x)
        
        # Original matrices
        Q = self.solenoidal.Q
        L_orig = jnp.tril(self.dissipative.W)
        Gamma_orig = L_orig @ L_orig.T
        
        # Masking
        M = self.hull.get_topology_mask()
        Q_masked = Q * M
        Gamma_masked = Gamma_orig * M
        
        # Safely compute S via eigendecomposition to restore positive-definiteness
        evals, evecs = jnp.linalg.eigh(Gamma_masked + self.epsilon * jnp.eye(self.d_state))
        evals = jnp.maximum(evals, 0.0)
        S = evecs @ jnp.diag(jnp.sqrt(evals))
        
        # Execute Thermostat with masked matrices
        x_next = self.thermostat(
            x=x,
            grad_E=grad_E,
            Q=Q_masked,
            L=S,
            dt=dt,
            key=key,
            omega_ext=omega_ext
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
        key: jax.random.PRNGKey,
        D_s: jax.Array | None = None,
        epsilon: float = 1e-4
    ):
        self.hull = MarkovHull(d_internal, d_sensory, d_active, d_external, D_s=D_s)
        d_state = self.hull.d_state
        
        k1, k2, k3 = jax.random.split(key, 3)
        self.ebm = PrecisionWeightedEBM(
            d_state=d_state,
            hidden_size=ebm_hidden_size,
            depth=ebm_depth,
            key=k1
        )
        self.solenoidal = SolenoidalFlow(d_state=d_state, key=k2)
        self.dissipative = DissipativeFriction(d_state=d_state, key=k3)
        self.thermostat = Thermostat(temperature=temperature)
        
        masked_factor = MaskedThermoFlowFactor(
            ebm=self.ebm,
            solenoidal=self.solenoidal,
            dissipative=self.dissipative,
            thermostat=self.thermostat,
            hull=self.hull,
            d_state=d_state,
            epsilon=epsilon
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

    def __call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float) -> jax.Array:
        """
        Executes the unrolled simulation over n_steps.
        """
        return self.thermalizer(key, x_init, dt)

    def forced_unroll(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float, seq: jax.Array | None = None, omega_seq: jax.Array | None = None) -> jax.Array:
        """
        Executes the unrolled simulation over an external sequence.
        """
        return self.forced_thermalizer(key, x_init, dt, seq=seq, omega_seq=omega_seq)

    def extract_internal_state(self, x: jax.Array) -> dict:
        """
        Extracts the internal partitions of a given state vector.
        """
        return self.hull.partition(x)
