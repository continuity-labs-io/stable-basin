import jax
import jax.numpy as jnp
from src.echo.architecture.markov_hull import MarkovHull
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph

def test_hull_degradation():
    d_i, d_s, d_a, d_e = 4, 3, 2, 1
    x = jnp.arange(10, dtype=jnp.float32)
    
    # Test 1: D_s = None
    hull_none = MarkovHull(d_i, d_s, d_a, d_e, D_s=None)
    x_obs_none = hull_none.apply_sensory_degradation(x)
    assert jnp.allclose(x, x_obs_none)
    
    # Test 2: D_s = zeros
    D_s = jnp.zeros((d_s, d_s))
    hull_zero = MarkovHull(d_i, d_s, d_a, d_e, D_s=D_s)
    x_obs_zero = hull_zero.apply_sensory_degradation(x)
    
    part_orig = hull_none.partition(x)
    part_obs = hull_zero.partition(x_obs_zero)
    
    assert jnp.allclose(part_orig["internal"], part_obs["internal"])
    assert jnp.allclose(part_orig["active"], part_obs["active"])
    assert jnp.allclose(part_orig["external"], part_obs["external"])
    
    assert jnp.allclose(part_obs["sensory"], jnp.zeros(d_s))


def test_observer_gradient_blindness():
    key = jax.random.PRNGKey(0)
    d_i, d_s, d_a, d_e = 4, 3, 2, 1
    D_s = jnp.zeros((d_s, d_s))
    
    obs = MarkovBlanketObserver(
        d_internal=d_i,
        d_sensory=d_s,
        d_active=d_a,
        d_external=d_e,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=1,
        temperature=1.0,
        key=key,
        D_s=D_s
    )
    
    x = jax.random.normal(key, (obs.hull.d_state,))
    
    def energy_fn(state):
        state_obs = obs.hull.apply_sensory_degradation(state)
        e, _ = obs.ebm(state_obs)
        return e
        
    grad_x = jax.grad(energy_fn)(x)
    grad_part = obs.hull.partition(grad_x)
    
    assert jnp.allclose(grad_part["sensory"], jnp.zeros(d_s))


def test_hierarchical_surprisal_blindness():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    d_i, d_s, d_a, d_e = 4, 3, 2, 1
    D_s_zero = jnp.zeros((d_s, d_s))
    
    micro = MarkovBlanketObserver(
        d_internal=d_i,
        d_sensory=d_s,
        d_active=d_a,
        d_external=d_e,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=1,
        temperature=1.0,
        key=k1,
        D_s=D_s_zero
    )
    
    macro = MarkovBlanketObserver(
        d_internal=d_i,
        d_sensory=d_s,
        d_active=d_a,
        d_external=d_e,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=1,
        temperature=1.0,
        key=k2
    )
    
    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    factor = graph.forced_thermalizer.flow_factor
    
    x_u = jax.random.normal(k1, (micro.hull.d_state,))
    x_m = jax.random.normal(k2, (macro.hull.d_state,))
    
    def joint_energy_fn(x_u_val, x_m_val):
        x_u_obs = factor.micro_hull.apply_sensory_degradation(x_u_val)
        x_m_obs = factor.macro_hull.apply_sensory_degradation(x_m_val)
        
        E_micro, _ = factor.micro_ebm(x_u_obs)
        E_macro, Pi_macro = factor.macro_ebm(x_m_obs)
        
        belief = factor.W_down(x_m_obs)
        diff = x_u_obs - belief
        
        diff_proj = factor.W_down.weight.T @ diff
        penalty = 0.5 * diff_proj.T @ Pi_macro @ diff_proj
        
        return E_micro + E_macro + penalty
        
    grad_u = jax.grad(joint_energy_fn, argnums=0)(x_u, x_m)
    grad_u_part = factor.micro_hull.partition(grad_u)
    
    assert jnp.allclose(grad_u_part["sensory"], jnp.zeros(d_s))
