import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from src.echo.primitives.ebm import PrecisionWeightedEBM

def test_precision_weighted_ebm_shapes_and_constraints():
    """
    Test the basic shape constraints and properties (Energy is scalar,
    Precision is SPD) of the PrecisionWeightedEBM.
    """
    # ARRANGE
    d_state = 4
    hidden_size = 16
    depth = 2
    epsilon = 1e-4
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(42))
    
    ebm = PrecisionWeightedEBM(
        d_state=d_state,
        hidden_size=hidden_size,
        depth=depth,
        key=key_mod,
        epsilon=epsilon
    )
    
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    # ACT
    energy, precision = ebm(x)
    
    # ASSERT
    assert energy.shape == (), "Energy output must be a scalar."
    assert not jnp.isnan(energy), "Energy contains NaNs."
    
    assert precision.shape == (d_state, d_state), "Precision output shape mismatch."
    assert not jnp.any(jnp.isnan(precision)), "Precision contains NaNs."
    
    # Assert symmetric
    assert jnp.allclose(precision, precision.T, atol=1e-5), "Precision matrix is not symmetric."
    
    # Assert positive-definite via eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(precision)
    assert jnp.all(eigenvalues >= epsilon - 1e-6), "Precision matrix has eigenvalues smaller than epsilon."

def test_precision_weighted_ebm_first_derivative():
    """
    Test that the first derivative (The Force) can be computed
    without NaNs and returns the correct shape.
    """
    # ARRANGE
    d_state = 4
    hidden_size = 16
    depth = 2
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(1337))
    
    ebm = PrecisionWeightedEBM(
        d_state=d_state,
        hidden_size=hidden_size,
        depth=depth,
        key=key_mod
    )
    
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    def energy_fn(input_x):
        e, _ = ebm(input_x)
        return e
    
    # ACT
    grad_E = jax.grad(energy_fn)(x)
    
    # ASSERT
    assert grad_E.shape == (d_state,), "Gradient shape mismatch."
    assert not jnp.any(jnp.isnan(grad_E)), "Gradient contains NaNs."

def test_precision_weighted_ebm_second_derivative():
    """
    Test that the second derivative (The Curvature) can be computed
    without NaNs and returns the correct shape, verifying smooth activations.
    """
    # ARRANGE
    d_state = 4
    hidden_size = 16
    depth = 2
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(999))
    
    ebm = PrecisionWeightedEBM(
        d_state=d_state,
        hidden_size=hidden_size,
        depth=depth,
        key=key_mod
    )
    
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    def energy_fn(input_x):
        e, _ = ebm(input_x)
        return e
    
    # ACT
    hessian_E = jax.hessian(energy_fn)(x)
    
    # ASSERT
    assert hessian_E.shape == (d_state, d_state), "Hessian shape mismatch."
    assert not jnp.any(jnp.isnan(hessian_E)), "Hessian contains NaNs (likely due to non-smooth activation like ReLU)."
    # Also verify Hessian is symmetric
    assert jnp.allclose(hessian_E, hessian_E.T, atol=1e-5), "Hessian should be symmetric."

def test_precision_weighted_ebm_jit_and_vmap():
    """
    Test that the module can be successfully JIT-compiled and batched.
    """
    # ARRANGE
    d_state = 4
    hidden_size = 16
    depth = 2
    batch_size = 8
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(1234))
    
    ebm = PrecisionWeightedEBM(
        d_state=d_state,
        hidden_size=hidden_size,
        depth=depth,
        key=key_mod
    )
    
    x_batch = jax.random.normal(key_x, (batch_size, d_state), dtype=jnp.float32)
    
    # ACT
    @eqx.filter_jit
    def batched_forward(mod, x_b):
        return eqx.filter_vmap(mod)(x_b)
        
    energy_batch, precision_batch = batched_forward(ebm, x_batch)
    
    # ASSERT
    assert energy_batch.shape == (batch_size,), "Batched energy shape mismatch."
    assert precision_batch.shape == (batch_size, d_state, d_state), "Batched precision shape mismatch."
    assert not jnp.any(jnp.isnan(energy_batch)), "Batched energy contains NaNs."
    assert not jnp.any(jnp.isnan(precision_batch)), "Batched precision contains NaNs."
