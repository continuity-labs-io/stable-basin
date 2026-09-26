import jax
import jax.numpy as jnp
import pytest
from src.echo.primitives.ebm_structured import StructuredPrecisionEBM

def test_structured_precision_ebm_shapes_and_symmetry():
    # ARRANGE
    key = jax.random.PRNGKey(42)
    d_state = 10
    hidden_size = 32
    depth = 2
    rank = 4
    epsilon = 1e-4
    ebm = StructuredPrecisionEBM(
        d_state=d_state, 
        hidden_size=hidden_size, 
        depth=depth, 
        key=key, 
        rank=rank, 
        epsilon=epsilon
    )
    x = jax.random.normal(key, (d_state,))

    # ACT
    energy, precision = ebm(x)

    # ASSERT
    # Shape consistency
    assert energy.shape == ()
    assert precision.shape == (d_state, d_state)
    
    # Mathematical invariant: Symmetry
    assert jnp.allclose(precision, precision.T, atol=1e-5), "Precision matrix must be symmetric."

def test_structured_precision_ebm_positive_definite():
    # ARRANGE
    key = jax.random.PRNGKey(123)
    d_state = 15
    hidden_size = 64
    depth = 2
    rank = 3
    epsilon = 1e-3
    ebm = StructuredPrecisionEBM(
        d_state=d_state, 
        hidden_size=hidden_size, 
        depth=depth, 
        key=key, 
        rank=rank, 
        epsilon=epsilon
    )
    x = jax.random.normal(key, (d_state,))

    # ACT
    _, precision = ebm(x)
    
    # ASSERT
    # Mathematical invariant: Positive Definiteness (Eigenvalues strictly > 0)
    # Since precision = diag(v) + U U^T where v = softplus(.) + epsilon > 0,
    # the matrix must be strictly positive definite.
    eigenvalues = jnp.linalg.eigvalsh(precision)
    assert jnp.all(eigenvalues > 0), "Precision matrix must be positive definite."
    # The smallest eigenvalue should be at least epsilon
    assert jnp.min(eigenvalues) >= epsilon * 0.99, f"Smallest eigenvalue should be >= epsilon ({epsilon})"

def test_structured_precision_ebm_gradient_stability():
    # ARRANGE
    key = jax.random.PRNGKey(99)
    d_state = 8
    hidden_size = 16
    depth = 2
    rank = 2
    ebm = StructuredPrecisionEBM(
        d_state=d_state, 
        hidden_size=hidden_size, 
        depth=depth, 
        key=key, 
        rank=rank
    )
    x = jnp.zeros((d_state,))

    # ACT
    def energy_fn(x_in):
        e, _ = ebm(x_in)
        return e
    
    grad_fn = jax.grad(energy_fn)
    grad_x = grad_fn(x)

    # ASSERT
    # Mathematical invariant: Gradient stability
    assert grad_x.shape == (d_state,)
    assert jnp.all(jnp.isfinite(grad_x)), "Energy gradients must be finite."
