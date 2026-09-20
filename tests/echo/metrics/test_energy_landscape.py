import jax
import jax.numpy as jnp
import pytest

from src.echo.metrics.energy_landscape import calculate_curvature, batch_calculate_curvature

def test_calculate_curvature_shapes():
    d_state = 10
    x = jnp.ones(d_state)
    
    # Dummy energy function
    def energy_fn(state):
        return jnp.sum(state ** 2)
        
    metrics = calculate_curvature(energy_fn, x)
    
    assert "hessian_trace" in metrics
    assert "eigenvalues" in metrics
    assert "hessian_rank" in metrics
    assert "hessian_nullity" in metrics
    
    assert metrics["hessian_trace"].shape == ()
    assert metrics["eigenvalues"].shape == (d_state,)
    assert metrics["hessian_rank"].shape == ()
    assert metrics["hessian_nullity"].shape == ()
    
    assert not jnp.isnan(metrics["hessian_trace"])
    assert not jnp.any(jnp.isnan(metrics["eigenvalues"]))

def test_batch_calculate_curvature_shapes():
    d_state = 10
    seq_len = 100
    x_seq = jnp.ones((seq_len, d_state))
    
    def energy_fn(state):
        return jnp.sum(state ** 2)
        
    metrics = batch_calculate_curvature(energy_fn, x_seq)
    
    assert metrics["hessian_trace"].shape == (seq_len,)
    assert metrics["eigenvalues"].shape == (seq_len, d_state)
    assert metrics["hessian_rank"].shape == (seq_len,)
    assert metrics["hessian_nullity"].shape == (seq_len,)

def test_convexity_perfect_bowl():
    d_state = 4
    k = 3.0
    x = jax.random.normal(jax.random.PRNGKey(0), (d_state,))
    
    def energy_fn(state):
        return 0.5 * k * jnp.sum(state ** 2)
        
    metrics = calculate_curvature(energy_fn, x)
    
    # The Hessian of 0.5 * k * sum(x^2) is exactly k * I.
    # Therefore, the trace should be exactly k * d_state.
    expected_trace = k * d_state
    
    assert jnp.allclose(metrics["hessian_trace"], expected_trace, atol=1e-5)
    
    # Check eigenvalues are all exactly k
    assert jnp.allclose(metrics["eigenvalues"], jnp.ones(d_state) * k, atol=1e-5)
    
    # Rank should be full
    assert metrics["hessian_rank"] == d_state
    assert metrics["hessian_nullity"] == 0
