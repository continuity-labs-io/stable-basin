import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph

def _create_graph(key):
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Micro observer (d_state=4) -> d_internal=1, d_sensory=1, d_active=1, d_external=1
    micro = MarkovBlanketObserver(1, 1, 1, 1, ebm_hidden_size=8, ebm_depth=2, n_steps=3, temperature=1.0, key=k1)
    
    # Macro observer (d_state=6) -> d_internal=2, d_sensory=1, d_active=1, d_external=2
    macro = MarkovBlanketObserver(2, 1, 1, 2, ebm_hidden_size=8, ebm_depth=2, n_steps=3, temperature=1.0, key=k2)
    
    graph = PredictiveCodingGraph(micro, macro, n_steps=3, key=k3)
    return graph

def test_hierarchy_execution():
    """
    Instantiates PredictiveCodingGraph and verifies execution without NaNs.
    """
    key = jax.random.PRNGKey(42)
    k_init, k_run, k_x1, k_x2 = jax.random.split(key, 4)
    
    graph = _create_graph(k_init)
    
    x_micro = jax.random.normal(k_x1, (4,), dtype=jnp.float32)
    x_macro = jax.random.normal(k_x2, (6,), dtype=jnp.float32)
    
    out = graph(k_run, x_micro, x_macro, dt=0.01)
    
    # Out may be a trajectory of (3, 10) or just final state (10,) depending on feedback_porting
    # But TorxThermalizer in the previous tests outputted either. We just assert no NaNs.
    assert not jnp.any(jnp.isnan(out))
    
    if out.ndim > 1:
        assert out.shape == (3, 10)
    else:
        assert out.shape == (10,)

def test_hierarchy_cross_talk_acid_test():
    """
    The BPTT Cross-Talk Proof (The Acid Test):
    Proves bidirectional coupling via gradients across unrolled time steps.
    """
    key = jax.random.PRNGKey(123)
    k_init, k_run, k_x1, k_x2 = jax.random.split(key, 4)
    
    graph = _create_graph(k_init)
    
    x_micro = jax.random.normal(k_x1, (4,), dtype=jnp.float32)
    x_macro = jax.random.normal(k_x2, (6,), dtype=jnp.float32)
    
    dt = 0.01
    
    # Test 1: Bottom-Up Surprisal
    @eqx.filter_value_and_grad
    def macro_loss_fn(x_u):
        out = graph(k_run, x_u, x_macro, dt)
        if out.ndim > 1:
            out = out[-1]
        macro_out = out[4:]
        return jnp.sum(macro_out)
        
    loss_m, grad_micro = macro_loss_fn(x_micro)
    
    assert not jnp.isnan(loss_m)
    assert not jnp.any(jnp.isnan(grad_micro))
    assert jnp.linalg.norm(grad_micro) > 0.0, "Bottom-Up Surprisal failed! Micro state did not perturb future Macro state."
    
    # Test 2: Top-Down Enslavement
    @eqx.filter_value_and_grad
    def micro_loss_fn(x_m):
        out = graph(k_run, x_micro, x_m, dt)
        if out.ndim > 1:
            out = out[-1]
        micro_out = out[:4]
        return jnp.sum(micro_out)
        
    loss_u, grad_macro = micro_loss_fn(x_macro)
    
    assert not jnp.isnan(loss_u)
    assert not jnp.any(jnp.isnan(grad_macro))
    assert jnp.linalg.norm(grad_macro) > 0.0, "Top-Down Enslavement failed! Macro belief did not alter Micro trajectory."

def test_hierarchy_jit():
    """
    Asserts the PredictiveCodingGraph compiles seamlessly with jax.jit.
    """
    key = jax.random.PRNGKey(999)
    k_init, k_run, k_x1, k_x2 = jax.random.split(key, 4)
    
    graph = _create_graph(k_init)
    
    x_micro = jax.random.normal(k_x1, (4,), dtype=jnp.float32)
    x_macro = jax.random.normal(k_x2, (6,), dtype=jnp.float32)
    
    @eqx.filter_jit
    def jitted_call(g, k, xu, xm):
        return g(k, xu, xm, 0.01)
        
    out = jitted_call(graph, k_run, x_micro, x_macro)
    assert not jnp.any(jnp.isnan(out))
