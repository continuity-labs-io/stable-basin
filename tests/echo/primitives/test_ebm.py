import pytest
import jax
import jax.numpy as jnp
from src.echo.primitives.ebm import GaussianEBM

def test_gaussian_ebm():
    """
    Test GaussianEBM initialization and forward pass mathematically.
    """
    d_state = 4
    key = jax.random.PRNGKey(42)
    
    # ARRANGE
    ebm = GaussianEBM(d_state=d_state, hidden_size=8, depth=2, key=key)
    x = jnp.array([1.0, -0.5, 2.0, 0.0])
    
    # ACT
    energy, precision = ebm(x)
    
    # ASSERT
    # 1. Output shapes and types
    assert energy.shape == ()
    assert precision.shape == (d_state, d_state)
    
    # 2. Strict Positive-Definiteness of Precision Matrix
    # All eigenvalues should be strictly positive
    eigenvalues = jnp.linalg.eigvals(precision)
    assert jnp.all(jnp.real(eigenvalues) > 0)
    
    # 3. Energy Positivity (since it's a quadratic form + constant)
    assert energy >= 0.0
    
    # 4. Energy mathematically matches 0.5 * diff^T * Pi * diff
    diff = x - ebm.mu
    expected_energy = 0.5 * jnp.dot(diff, jnp.dot(precision, diff))
    assert jnp.isclose(energy, expected_energy, atol=1e-5)
