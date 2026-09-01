import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from src.echo.architecture.markov_hull import MarkovHull
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.clinic.interventions import DigitalTwinAnnealer, DigitalTwinInterrogator
from src.echo.physics.dissipative import DissipativeFriction
from src.echo.primitives.ebm import PrecisionWeightedEBM

@pytest.fixture
def degraded_graph():
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Very small dimensions for testing
    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 2
    
    # Degraded sensory mask
    D_s = jnp.zeros((d_sensory, d_sensory))
    
    micro_obs = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=1,
        temperature=0.0,
        key=k1,
        D_s=D_s
    )
    
    macro_obs = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=1,
        temperature=0.0,
        key=k2
    )
    
    graph = PredictiveCodingGraph(
        micro_observer=micro_obs,
        macro_observer=macro_obs,
        n_steps=1,
        key=k3
    )
    return graph

def get_weight_norms(graph: PredictiveCodingGraph):
    def get_norms(x):
        if isinstance(x, DissipativeFriction):
            return jnp.linalg.norm(x.W)
        if isinstance(x, PrecisionWeightedEBM):
            return jnp.linalg.norm(x.precision_head.weight)
        return None
        
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            get_norms, 
            graph, 
            is_leaf=lambda x: isinstance(x, (DissipativeFriction, PrecisionWeightedEBM))
        )
    )
    # Filter out Nones
    return [l for l in leaves if l is not None]

def test_digital_twin_annealer(degraded_graph):
    annealer = DigitalTwinAnnealer()
    
    gamma_boost = 1.5
    pi_boost = 2.0
    
    norms_a = get_weight_norms(degraded_graph)
    
    # Act
    annealed_graph = annealer.anneal_twin(degraded_graph, gamma_boost=gamma_boost, pi_boost=pi_boost)
    
    norms_b = get_weight_norms(annealed_graph)
    norms_a_after = get_weight_norms(degraded_graph)
    
    # Assert immutability of Twin A
    for na, na_after in zip(norms_a, norms_a_after):
        assert jnp.allclose(na, na_after), "Original graph mutated in place!"
        
    # Assert Twin B is boosted
    assert len(norms_a) > 0
    assert len(norms_a) == len(norms_b)
    for na, nb in zip(norms_a, norms_b):
        assert nb > na, "Annealed graph weights should be larger than original."

def test_digital_twin_interrogator(degraded_graph):
    interrogator = DigitalTwinInterrogator()
    
    factor = degraded_graph.thermalizer.graph.sites[0].factor.base
    d_micro = factor.d_micro
    d_macro = factor.d_macro
    
    x_micro = jnp.ones(d_micro)
    x_macro = jnp.zeros(d_macro)
    
    q_ext_pulse = jnp.ones(d_micro) * 0.1
    
    res = interrogator.ping_and_measure(degraded_graph, x_micro, x_macro, q_ext_pulse)
    
    assert isinstance(res, dict)
    assert "micro_surprisal" in res
    assert "macro_surprisal" in res
    assert "discordance" in res
    
    assert not jnp.isnan(res["discordance"])
    assert res["discordance"] >= 0.0

def test_interrogator_blindness(degraded_graph):
    interrogator = DigitalTwinInterrogator()
    
    factor = degraded_graph.thermalizer.graph.sites[0].factor.base
    d_micro = factor.d_micro
    d_macro = factor.d_macro
    
    x_micro = jnp.ones(d_micro)
    x_macro = jnp.zeros(d_macro)
    q_ext_pulse = jnp.ones(d_micro) * 0.5
    
    # Mock a blind macro gradient (all zeros) by overriding the joint_energy_fn
    original_fn = factor.joint_energy_fn
    
    def mocked_joint_energy_fn(self, x_u, x_m):
        # Micro has gradient (return sum of squares), Macro has zero gradient (ignore x_m)
        return jnp.sum(x_u ** 2)
        
    from unittest.mock import patch
    with patch('src.echo.architecture.hierarchy.HierarchicalThermoFlowFactor.joint_energy_fn', new=mocked_joint_energy_fn):
        res = interrogator.ping_and_measure(degraded_graph, x_micro, x_macro, q_ext_pulse)
        
        # Macro gradient should be exactly zero
        assert jnp.allclose(res["macro_surprisal"], 0.0)
        # Discordance should therefore be exactly zero
        assert jnp.allclose(res["discordance"], 0.0)
