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
from src.echo.architecture.observer import MarkovBlanketObserver


class HierarchicalThermoFlowFactor(torx.factor.AbstractReferenceFactor):
    """
    Custom Torx factor that defines a joint free energy over a Micro and a Macro 
    Markov Blanket Observer, enabling automatic message passing via gradient flow.
    """
    micro_hull: MarkovHull
    macro_hull: MarkovHull
    
    micro_ebm: PrecisionWeightedEBM
    macro_ebm: PrecisionWeightedEBM
    
    micro_solenoidal: SolenoidalFlow
    macro_solenoidal: SolenoidalFlow
    
    micro_dissipative: DissipativeFriction
    macro_dissipative: DissipativeFriction
    
    micro_thermostat: Thermostat
    macro_thermostat: Thermostat
    
    W_down: eqx.nn.Linear
    
    d_micro: int = eqx.field(static=True)
    d_macro: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(
        self,
        micro_hull: MarkovHull,
        macro_hull: MarkovHull,
        micro_ebm: PrecisionWeightedEBM,
        macro_ebm: PrecisionWeightedEBM,
        micro_solenoidal: SolenoidalFlow,
        macro_solenoidal: SolenoidalFlow,
        micro_dissipative: DissipativeFriction,
        macro_dissipative: DissipativeFriction,
        micro_thermostat: Thermostat,
        macro_thermostat: Thermostat,
        W_down: eqx.nn.Linear,
        d_micro: int,
        d_macro: int,
        epsilon: float = 1e-4
    ):
        self.micro_hull = micro_hull
        self.macro_hull = macro_hull
        
        self.micro_ebm = micro_ebm
        self.macro_ebm = macro_ebm
        
        self.micro_solenoidal = micro_solenoidal
        self.macro_solenoidal = macro_solenoidal
        
        self.micro_dissipative = micro_dissipative
        self.macro_dissipative = macro_dissipative
        
        self.micro_thermostat = micro_thermostat
        self.macro_thermostat = macro_thermostat
        
        self.W_down = W_down
        self.d_micro = d_micro
        self.d_macro = d_macro
        self.epsilon = epsilon
        
        self.input_ports = {
            "x": jax.ShapeDtypeStruct((d_micro + d_macro,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "omega_ext": jax.ShapeDtypeStruct((d_micro + d_macro,), jnp.float32),
            "q_ext": jax.ShapeDtypeStruct((d_micro + d_macro,), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((d_micro + d_macro,), jnp.float32)

    def init_params(self, key):
        return {}

    def joint_energy_fn(self, x_u, x_m):
        """
        Computes the Joint Free Energy (F) of the hierarchical system.
        This represents the total energy landscape that both the micro and macro 
        states physically flow down. It is composed of the independent endogenous 
        energies of each level, plus a predictive coding coupling penalty that 
        binds them together.
        
        Args:
            x_u: The full state vector of the micro-level observer.
            x_m: The full state vector of the macro-level observer.
            
        Returns:
            F: A scalar representing the total Joint Free Energy evaluated at (x_u, x_m).
        """
        x_u_obs = self.micro_hull.apply_sensory_degradation(x_u)
        x_m_obs = self.macro_hull.apply_sensory_degradation(x_m)
        
        E_micro, _ = self.micro_ebm(x_u_obs)
        E_macro, Pi_macro = self.macro_ebm(x_m_obs)
        
        belief = self.W_down(x_m_obs)
        diff = x_u_obs - belief
        
        # Project the prediction error (diff) back up into the macro latent space
        diff_proj = self.W_down.weight.T @ diff
        
        # Precision-Weighted Prediction Error (Mahalanobis Distance):
        # Pi_macro is the precision (inverse variance) matrix of the macro prior.
        # This computes a quadratic penalty: 1/2 * e^T * Pi * e.
        # - High precision (large Pi) acts as a stiff spring, heavily penalizing 
        #   deviations and forcing the micro state to snap to the macro belief.
        # - Low precision (small Pi) acts as a loose spring, allowing the micro 
        #   state to freely fluctuate without being dragged by the macro level.
        penalty = 0.5 * diff_proj.T @ Pi_macro @ diff_proj
        
        F = E_micro + E_macro + penalty
        return F

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        x = inputs["x"]
        dt = inputs["dt"]
        omega_ext = inputs.get("omega_ext", jnp.zeros(self.d_micro + self.d_macro, dtype=jnp.float32))
        q_ext = inputs.get("q_ext", jnp.zeros(self.d_micro + self.d_macro, dtype=jnp.float32))
        
        # a) Split input and forces
        x_micro = x[:self.d_micro]
        x_macro = x[self.d_micro:]
        
        omega_micro = omega_ext[:self.d_micro]
        omega_macro = omega_ext[self.d_micro:]
        
        q_micro = q_ext[:self.d_micro]
        q_macro = q_ext[self.d_micro:]
        
        # c) Compute gradients simultaneously
        grad_micro, grad_macro = jax.grad(self.joint_energy_fn, argnums=(0, 1))(x_micro, x_macro)
        
        # d) Apply respective Hull masks
        M_micro = self.micro_hull.get_topology_mask()
        Q_micro_masked = self.micro_solenoidal.Q * M_micro
        L_micro_orig = jnp.tril(self.micro_dissipative.W)
        Gamma_micro_orig = L_micro_orig @ L_micro_orig.T
        Gamma_micro_masked = Gamma_micro_orig * M_micro
        
        M_macro = self.macro_hull.get_topology_mask()
        Q_macro_masked = self.macro_solenoidal.Q * M_macro
        L_macro_orig = jnp.tril(self.macro_dissipative.W)
        Gamma_macro_orig = L_macro_orig @ L_macro_orig.T
        Gamma_macro_masked = Gamma_macro_orig * M_macro
        
        # e) Compute safe diffusion matrix S for both
        evals_u, evecs_u = jnp.linalg.eigh(Gamma_micro_masked + self.epsilon * jnp.eye(self.d_micro))
        evals_u = jnp.maximum(evals_u, 0.0)
        S_micro = evecs_u @ jnp.diag(jnp.sqrt(evals_u))
        
        evals_m, evecs_m = jnp.linalg.eigh(Gamma_macro_masked + self.epsilon * jnp.eye(self.d_macro))
        evals_m = jnp.maximum(evals_m, 0.0)
        S_macro = evecs_m @ jnp.diag(jnp.sqrt(evals_m))
        
        # f) Execute Thermostat steps independently
        k_micro, k_macro = jax.random.split(key, 2)
        
        x_micro_next = self.micro_thermostat(
            x=x_micro,
            grad_E=grad_micro,
            Q=Q_micro_masked,
            L=S_micro,
            dt=dt,
            key=k_micro,
            omega_ext=omega_micro,
            q_ext=q_micro
        )
        
        x_macro_next = self.macro_thermostat(
            x=x_macro,
            grad_E=grad_macro,
            Q=Q_macro_masked,
            L=S_macro,
            dt=dt,
            key=k_macro,
            omega_ext=omega_macro,
            q_ext=q_macro
        )
        
        # g) Concatenate and return
        x_next = jnp.concatenate([x_micro_next, x_macro_next])
        
        if return_aux:
            return x_next, None
        return x_next


class PredictiveCodingGraph(eqx.Module):
    """
    Couples a Micro and a Macro Markov Blanket Observer into a nested 
    hierarchical predictive coding network.
    """
    W_down: eqx.nn.Linear
    thermalizer: TorxThermalizer
    forced_thermalizer: ForcedTorxThermalizer
    d_micro: int = eqx.field(static=True)
    d_macro: int = eqx.field(static=True)
    
    def __init__(self, micro_observer: MarkovBlanketObserver, macro_observer: MarkovBlanketObserver, n_steps: int, key: jax.random.PRNGKey):
        self.d_micro = micro_observer.hull.d_state
        self.d_macro = macro_observer.hull.d_state
        
        self.W_down = eqx.nn.Linear(self.d_macro, self.d_micro, use_bias=False, key=key)
        
        factor = HierarchicalThermoFlowFactor(
            micro_hull=micro_observer.hull,
            macro_hull=macro_observer.hull,
            micro_ebm=micro_observer.ebm,
            macro_ebm=macro_observer.ebm,
            micro_solenoidal=micro_observer.solenoidal,
            macro_solenoidal=macro_observer.solenoidal,
            micro_dissipative=micro_observer.dissipative,
            macro_dissipative=macro_observer.dissipative,
            micro_thermostat=micro_observer.thermostat,
            macro_thermostat=macro_observer.thermostat,
            W_down=self.W_down,
            d_micro=self.d_micro,
            d_macro=self.d_macro
        )
        
        d_state = self.d_micro + self.d_macro
        self.thermalizer = TorxThermalizer(
            flow_factor=factor,
            n_steps=n_steps,
            d_state=d_state
        )
        self.forced_thermalizer = ForcedTorxThermalizer(
            flow_factor=factor,
            d_state=d_state,
            injection_start_idx=micro_observer.hull.d_internal
        )
        
    def __call__(self, key: jax.random.PRNGKey, x_micro_init: jax.Array, x_macro_init: jax.Array, dt: float) -> jax.Array:
        """
        Executes the unrolled joint simulation over n_steps.
        """
        x_init = jnp.concatenate([x_micro_init, x_macro_init])
        return self.thermalizer(key, x_init, dt)

    def forced_unroll(self, key: jax.random.PRNGKey, x_micro_init: jax.Array, x_macro_init: jax.Array, dt: float, seq: jax.Array | None = None, omega_seq: jax.Array | None = None, q_gain: float = 0.0, q_mask: jax.Array | None = None) -> jax.Array:
        """
        Executes the unrolled joint simulation over an external sequence.
        """
        x_init = jnp.concatenate([x_micro_init, x_macro_init])
        return self.forced_thermalizer(key, x_init, dt, seq=seq, omega_seq=omega_seq, q_gain=q_gain, q_mask=q_mask)
