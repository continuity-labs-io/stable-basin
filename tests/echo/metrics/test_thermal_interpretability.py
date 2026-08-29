import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.metrics.thermal_interpretability import HessianCurvatureTracker


def test_hessian_curvature_tracker_single():
    """
    Instantiates tracker and asserts return dictionary shapes for a single state.
    """
    key = jax.random.PRNGKey(42)
    ebm = PrecisionWeightedEBM(d_state=4, hidden_size=16, depth=2, key=key)
    tracker = HessianCurvatureTracker(ebm)
    
    x = jax.random.normal(jax.random.PRNGKey(1), (4,), dtype=jnp.float32)
    metrics = tracker.calculate_curvature(x)
    
    assert "hessian_trace" in metrics
    assert "explicit_precision_trace" in metrics
    assert "eigenvalues" in metrics
    
    assert metrics["hessian_trace"].shape == ()
    assert metrics["explicit_precision_trace"].shape == ()
    assert metrics["eigenvalues"].shape == (4,)
    
    assert not jnp.isnan(metrics["hessian_trace"])
    assert not jnp.isnan(metrics["explicit_precision_trace"])
    assert not jnp.any(jnp.isnan(metrics["eigenvalues"]))


def test_hessian_curvature_tracker_batch():
    """
    Shape & VMAP Test: Asserts the vmap wrapper works safely on trajectories.
    """
    key = jax.random.PRNGKey(42)
    ebm = PrecisionWeightedEBM(d_state=4, hidden_size=16, depth=2, key=key)
    tracker = HessianCurvatureTracker(ebm)
    
    x_seq = jax.random.normal(jax.random.PRNGKey(1), (100, 4), dtype=jnp.float32)
    metrics = tracker.batch_calculate_curvature(x_seq)
    
    assert metrics["hessian_trace"].shape == (100,)
    assert metrics["explicit_precision_trace"].shape == (100,)
    assert metrics["eigenvalues"].shape == (100, 4)
    
    assert not jnp.any(jnp.isnan(metrics["hessian_trace"]))
    assert not jnp.any(jnp.isnan(metrics["explicit_precision_trace"]))
    assert not jnp.any(jnp.isnan(metrics["eigenvalues"]))


class PerfectBowlEBM(eqx.Module):
    """
    Dummy EBM that mimics PrecisionWeightedEBM signature but hardcodes 
    a perfect quadratic bowl energy landscape.
    """
    k: float = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    
    def __init__(self, k, d_state):
        self.k = k
        self.d_state = d_state
        
    def __call__(self, x):
        energy = 0.5 * self.k * jnp.sum(x**2)
        precision = jnp.eye(self.d_state)
        return energy, precision


def test_hessian_curvature_acid_test():
    """
    The Convexity Test (The Acid Test):
    Mathematically prove the Hessian is accurately computing physical curvature.
    """
    d_state = 4
    k = 3.0
    ebm = PerfectBowlEBM(k=k, d_state=d_state)
    tracker = HessianCurvatureTracker(ebm)
    
    x = jnp.array([1.0, -0.5, 2.0, 0.0])
    
    metrics = tracker.calculate_curvature(x)
    
    # Crucial Assertion: Hessian of 0.5 * k * sum(x^2) is exactly k * I.
    # Therefore, trace must be exactly k * d_state.
    expected_trace = k * d_state
    
    assert jnp.allclose(metrics["hessian_trace"], expected_trace), \
        f"Hessian Trace {metrics['hessian_trace']} did not match analytic expected trace {expected_trace}"
    
    # The eigenvalues should all be exactly k
    assert jnp.allclose(metrics["eigenvalues"], k), \
        f"Eigenvalues {metrics['eigenvalues']} are not all exactly {k}"

