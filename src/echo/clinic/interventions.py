import jax
import jax.numpy as jnp
import equinox as eqx

from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.primitives.ebm import PrecisionWeightedEBM

class DigitalTwinAnnealer(eqx.Module):
    """
    Mathematically restores a degraded Digital Twin to its optimal
    physical state without needing a population database.
    """
    def __init__(self):
        pass

    def anneal_twin(self, degraded_graph: PredictiveCodingGraph, gamma_boost: float = 1.5, pi_boost: float = 2.0) -> PredictiveCodingGraph:
        """
        Creates a new annealed counterfactual twin by multiplying the
        unconstrained friction weights by gamma_boost and the precision head
        weights by pi_boost.
        """
        def _map_fn(x):
            if isinstance(x, DissipativeFriction):
                return eqx.tree_at(lambda p: p.W, x, x.W * gamma_boost)
            if isinstance(x, PrecisionWeightedEBM):
                return eqx.tree_at(lambda p: p.precision_head.weight, x, x.precision_head.weight * pi_boost)
            return x

        # Apply transformations using tree_map with is_leaf
        annealed_graph = jax.tree_util.tree_map(
            _map_fn, 
            degraded_graph, 
            is_leaf=lambda x: isinstance(x, (DissipativeFriction, PrecisionWeightedEBM))
        )

        return annealed_graph

class DigitalTwinInterrogator(eqx.Module):
    """
    Measures "Silent Drift" (Blindness) by comparing micro-level
    surprisal to macro-level surprisal after an active ping.
    """
    def __init__(self):
        pass

    def ping_and_measure(self, graph: PredictiveCodingGraph, x_micro: jax.Array, x_macro: jax.Array, q_ext_pulse: jax.Array) -> dict:
        """
        Injects a bioelectric shock `q_ext_pulse` into `x_micro`, computes
        the joint free energy gradients, and measures the hierarchical discordance.
        
        Args:
            graph: A PredictiveCodingGraph containing the hierarchy's physics and energy landscape.
            x_micro: The full state vector of the micro-level observer.
            x_macro: The full state vector of the macro-level observer.
            q_ext_pulse: An exogenous perturbation vector representing a bioelectric shock applied to the micro level.
            
        Returns:
            dict: A dictionary containing:
                - "micro_surprisal": The L2 norm of the free energy gradient with respect to the shocked micro state.
                - "macro_surprisal": The L2 norm of the free energy gradient with respect to the macro state.
                - "discordance": The ratio of macro surprisal to micro surprisal (measuring Sensory Blindness).
        """
        x_micro_shocked = x_micro + q_ext_pulse
        
        # Extract the joint_energy_fn from the graph's logic
        # graph.thermalizer.flow_factor.base is the HierarchicalThermoFlowFactor
        factor = graph.thermalizer.graph.sites[0].factor.base
        joint_energy_fn = factor.joint_energy_fn
        
        # Compute gradients simultaneously
        grad_micro, grad_macro = jax.grad(joint_energy_fn, argnums=(0, 1))(x_micro_shocked, x_macro)
        
        micro_norm = jnp.linalg.norm(grad_micro)
        macro_norm = jnp.linalg.norm(grad_macro)
        
        discordance = macro_norm / (micro_norm + 1e-8)
        
        return {
            "micro_surprisal": micro_norm,
            "macro_surprisal": macro_norm,
            "discordance": discordance
        }
