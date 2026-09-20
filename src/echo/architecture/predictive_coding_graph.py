from jaxtyping import PRNGKeyArray
from jaxtyping import jaxtyped
from beartype import beartype
import jax
import jax.numpy as jnp
import equinox as eqx

from src.echo.architecture.markov_hull import MarkovHull
from src.echo.primitives.thermalizer import TorxThermalizer, ForcedTorxThermalizer
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchical_factor import HierarchicalThermoFlowFactor

class PredictiveCodingGraph(eqx.Module):
    """
    Couples a Micro and a Macro Markov Blanket Observer into a nested 
    hierarchical predictive coding network.
    """
    W_down: eqx.nn.Linear
    thermalizer: TorxThermalizer
    forced_thermalizer: ForcedTorxThermalizer
    flow_factor: HierarchicalThermoFlowFactor
    hull: MarkovHull
    d_micro: int = eqx.field(static=True)
    d_macro: int = eqx.field(static=True)

    def __init__(self, micro_observer: MarkovBlanketObserver, macro_observer: MarkovBlanketObserver, n_steps: int, key: PRNGKeyArray):
        # We need a hull that represents the concatenated state so apply_sensory_degradation works safely.
        # But EchoRunner's validation also uses model.hull to extract sensory dimensions.
        # We will create a mock hull for the whole graph.
        self.hull = MarkovHull(
            d_internal=micro_observer.hull.d_internal + macro_observer.hull.d_internal,
            d_sensory=micro_observer.hull.d_sensory,
            d_active=micro_observer.hull.d_active,
            d_external=micro_observer.hull.d_external + macro_observer.hull.d_state - macro_observer.hull.d_internal - micro_observer.hull.d_sensory - micro_observer.hull.d_active
        )
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
            d_macro=self.d_macro,
            use_micro_blanket=micro_observer.use_blanket_topology,
            use_macro_blanket=macro_observer.use_blanket_topology
        )
        
        self.flow_factor = factor
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

    @property
    def ebm(self):
        # Return a mock module that satisfies the EBM interface (returns energy, None)
        # and computes the joint free energy over the full concatenated state.
        # This allows EchoRunner's HessianCurvatureTracker to compute the full 46x46 Hessian.
        class JointEBM(eqx.Module):
            flow_factor: HierarchicalThermoFlowFactor
            d_micro: int
            
            @jaxtyped(typechecker=beartype)
            def __call__(self_, x):
                x_micro = x[:self_.d_micro]
                x_macro = x[self_.d_micro:]
                E = self_.flow_factor.joint_energy_fn(x_micro, x_macro)
                return E, jnp.eye(x.shape[0])
                
        return JointEBM(flow_factor=self.flow_factor, d_micro=self.d_micro)
        
    @jaxtyped(typechecker=beartype)
    def __call__(self, key: PRNGKeyArray, x_init: jax.Array, dt: float) -> jax.Array:
        """
        Executes the unrolled joint simulation over n_steps.
        """
        factor_params = self.thermalizer.graph.sites[0].factor.base.precompute()
        return self.thermalizer(key, x_init, dt, factor_params=factor_params)

    def forced_unroll(self, key: PRNGKeyArray, x_init: jax.Array, dt: float, seq: jax.Array | None = None, omega_seq: jax.Array | None = None, q_gain: float = 0.0, q_mask: jax.Array | None = None) -> jax.Array:
        """
        Executes the unrolled joint simulation over an external sequence.
        """
        factor_params = self.forced_thermalizer.flow_factor.precompute()
        return self.forced_thermalizer(key, x_init, dt, seq=seq, omega_seq=omega_seq, q_gain=q_gain, q_mask=q_mask, factor_params=factor_params)
