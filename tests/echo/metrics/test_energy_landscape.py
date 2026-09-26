import jax
import jax.numpy as jnp
import pytest

from src.echo.metrics.energy_landscape import calculate_curvature, batch_calculate_curvature


def test_calculate_curvature_shapes():
    d_state = 10
    x = jnp.ones(d_state)

    # Dummy energy function
    def energy_fn(state):
        return jnp.sum(state**2)

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
        return jnp.sum(state**2)

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
        return 0.5 * k * jnp.sum(state**2)

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

def test_hutchinson_hessian_trace():
    # ARRANGE
    from src.echo.metrics.energy_landscape import hutchinson_hessian_trace
    d_state = 10
    k_val = 3.0
    x = jax.random.normal(jax.random.PRNGKey(0), (d_state,))
    key = jax.random.PRNGKey(1)
    
    def energy_fn(state):
        return 0.5 * k_val * jnp.sum(state**2)
        
    # ACT
    trace_estimate = hutchinson_hessian_trace(energy_fn, x, key, n_probes=1000)
    
    # ASSERT
    expected_trace = k_val * d_state
    assert trace_estimate.shape == ()
    assert not jnp.isnan(trace_estimate)
    assert jnp.allclose(trace_estimate, expected_trace, atol=1e-5)


def test_batch_hutchinson_trace():
    # ARRANGE
    from src.echo.metrics.energy_landscape import batch_hutchinson_trace
    d_state = 10
    batch_size = 5
    k_val = 3.0
    x_seq = jax.random.normal(jax.random.PRNGKey(0), (batch_size, d_state))
    key = jax.random.PRNGKey(1)
    
    def energy_fn(state):
        return 0.5 * k_val * jnp.sum(state**2)
        
    # ACT
    trace_estimates = batch_hutchinson_trace(energy_fn, x_seq, key, n_probes=10)
    
    # ASSERT
    expected_trace = k_val * d_state
    assert trace_estimates.shape == (batch_size,)
    assert not jnp.any(jnp.isnan(trace_estimates))
    assert jnp.allclose(trace_estimates, expected_trace, atol=1e-5)


def test_curvature_over_states_hutchinson():
    # ARRANGE
    from src.echo.metrics.energy_landscape import curvature_over_states
    d_state = 5
    N = 12
    k_val = 2.0
    x_seq = jax.random.normal(jax.random.PRNGKey(0), (N, d_state))
    key = jax.random.PRNGKey(1)
    
    def energy_fn(state):
        return 0.5 * k_val * jnp.sum(state**2)
        
    # ACT
    res = curvature_over_states(
        energy_fn, 
        x_seq, 
        chunk_size=5, 
        estimator="hutchinson", 
        hutchinson_key=key, 
        hutchinson_probes=10
    )
    
    # ASSERT
    assert "hessian_trace" in res
    assert res["hessian_trace"].shape == (N,)
    assert jnp.allclose(res["hessian_trace"], k_val * d_state, atol=1e-5)


def test_hutchinson_estimator_accuracy():
    # ARRANGE
    from src.echo.metrics.energy_landscape import hutchinson_hessian_trace, hessian_trace
    d_state = 10
    x = jax.random.normal(jax.random.PRNGKey(0), (d_state,))
    key = jax.random.PRNGKey(1)
    
    A = jax.random.normal(jax.random.PRNGKey(2), (d_state, d_state))
    A = A + A.T  # symmetric
    def energy_fn(state):
        return 0.5 * jnp.dot(state, jnp.dot(A, state))
        
    # ACT
    exact_trace = hessian_trace(energy_fn, x)
    hutch_trace = hutchinson_hessian_trace(energy_fn, x, key, n_probes=2000)
    
    # ASSERT
    assert exact_trace.shape == ()
    assert hutch_trace.shape == ()
    assert jnp.allclose(hutch_trace, exact_trace, rtol=0.1, atol=1.5)


def test_hutchinson_fallback_rademacher():
    from src.echo.metrics.energy_landscape import hutchinson_hessian_trace
    import unittest.mock as mock
    
    d_state = 10
    x = jax.random.normal(jax.random.PRNGKey(0), (d_state,))
    key = jax.random.PRNGKey(1)
    
    def energy_fn(state):
        return jnp.sum(state**2)
        
    with mock.patch("jax.random.rademacher", side_effect=AttributeError, create=True) as mock_rad:
        # If we pretend hasattr returns False, we can test the fallback
        with mock.patch("src.echo.metrics.energy_landscape.hasattr", return_value=False):
            trace = hutchinson_hessian_trace(energy_fn, x, key, n_probes=10)
    assert not jnp.isnan(trace)


def test_curvature_over_states_edge_cases():
    from src.echo.metrics.energy_landscape import curvature_over_states
    
    def energy_fn(state):
        return jnp.sum(state**2)
        
    states = jnp.ones((5, 10))
    key = jax.random.PRNGKey(0)
    
    # Missing key for hutchinson
    with pytest.raises(ValueError, match="hutchinson_key must be provided"):
        curvature_over_states(energy_fn, states, estimator="hutchinson")
        
    # Unknown estimator
    with pytest.raises(ValueError, match="Unknown estimator"):
        curvature_over_states(energy_fn, states, estimator="unknown_algo")
        
    # exact_hessian (should run without error)
    res = curvature_over_states(energy_fn, states, estimator="exact_hessian", full_spectrum=False)
    assert "hessian_trace" in res
    assert len(res["hessian_trace"]) == 5


def test_scalar_energy_hull():
    from src.echo.metrics.energy_landscape import ScalarEnergy
    import equinox as eqx
    
    class DummyEBM(eqx.Module):
        def __call__(self, x):
            return x * 2.0, "pi"
            
    class DummyHull(eqx.Module):
        def apply_sensory_degradation(self, x):
            return x + 1.0
            
    # With hull and returning tuple
    energy = ScalarEnergy(DummyEBM(), DummyHull())
    x = jnp.array([1.0])
    assert jnp.allclose(energy(x), 4.0)
    
    # Without hull and returning single tensor
    class SimpleEBM(eqx.Module):
        def __call__(self, x):
            return x * 3.0
    energy2 = ScalarEnergy(SimpleEBM())
    assert jnp.allclose(energy2(x), 3.0)

