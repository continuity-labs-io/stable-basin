import pytest
import jax
import jax.numpy as jnp
from src.echo.primitives.ebm import GaussianEBM, IdentityPrecisionEBM

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


def test_identity_precision_ebm_happy_path():
    """
    Test IdentityPrecisionEBM initialization and forward pass mathematically.
    """
    d_state = 4
    key = jax.random.PRNGKey(42)
    
    # ARRANGE
    ebm = IdentityPrecisionEBM(d_state=d_state, hidden_size=8, depth=2, key=key)
    x = jnp.array([1.0, -0.5, 2.0, 0.0])
    
    # ACT
    energy, precision = ebm(x)
    
    # ASSERT
    # 1. Output shapes and types
    assert energy.shape == ()
    assert precision.shape == (d_state, d_state)
    
    # 2. Precision matrix is exactly Identity
    assert jnp.array_equal(precision, jnp.eye(d_state, dtype=jnp.float32))
    
    # 3. Energy Positivity
    assert energy >= 0.0


def test_identity_precision_ebm_unhappy_path():
    """
    Test IdentityPrecisionEBM constructor validation.
    """
    key = jax.random.PRNGKey(42)
    
    # d_state <= 0
    with pytest.raises(ValueError, match="d_state must be positive"):
        IdentityPrecisionEBM(d_state=0, hidden_size=8, depth=2, key=key)
        
    with pytest.raises(ValueError, match="d_state must be positive"):
        IdentityPrecisionEBM(d_state=-1, hidden_size=8, depth=2, key=key)
        
    # hidden_size <= 0
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        IdentityPrecisionEBM(d_state=4, hidden_size=0, depth=2, key=key)
        
    # depth < 0
    with pytest.raises(ValueError, match="depth must be non-negative"):
        IdentityPrecisionEBM(d_state=4, hidden_size=8, depth=-1, key=key)
