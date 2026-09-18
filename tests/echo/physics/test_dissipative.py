import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from src.echo.physics.dissipative import DissipativeFriction

def test_dissipative_friction_symmetry_and_pd():
    """
    Test that the Gamma matrix is symmetric and strictly positive definite
    by checking its eigenvalues.
    """
    # ARRANGE
    d_state = 64
    key = jax.random.PRNGKey(42)
    epsilon = 1e-4
    
    # ACT
    friction = DissipativeFriction(d_state=d_state, key=key, epsilon=epsilon)
    gamma = friction.Gamma
    
    # ASSERT
    # Assert symmetric
    assert jnp.allclose(gamma, gamma.T, atol=1e-5), "Gamma is not symmetric"
    
    # Assert positive-definite via eigenvalues
    eigenvalues, _ = jnp.linalg.eigh(gamma)
    assert jnp.all(eigenvalues >= epsilon - 1e-6), "Gamma has eigenvalues smaller than epsilon"

def test_dissipative_friction_quadratic_form():
    """
    Test that the quadratic form x^T Gamma x is strictly greater than 0
    for a non-zero vector x.
    """
    # ARRANGE
    d_state = 64
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(1337))
    epsilon = 1e-4
    friction = DissipativeFriction(d_state=d_state, key=key_mod, epsilon=epsilon)
    
    # Generate non-zero random vector x
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    # ACT
    gamma_x = friction(x)
    quadratic_form = jnp.dot(x, gamma_x)
    
    # ASSERT
    assert quadratic_form > 0.0, f"Quadratic form is not strictly positive, got {quadratic_form}"

def test_dissipative_friction_jit_and_vmap():
    """
    Test that the module can be successfully JIT-compiled and batched
    over a dimension.
    """
    # ARRANGE
    d_state = 64
    batch_size = 16
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(999))
    friction = DissipativeFriction(d_state=d_state, key=key_mod)
    x_batch = jax.random.normal(key_x, (batch_size, d_state), dtype=jnp.float32)
    
    # ACT
    # JIT compile the batched function
    @jax.jit
    def batched_forward(mod, x_b):
        return jax.vmap(mod)(x_b)
        
    out = batched_forward(friction, x_batch)
    
    # ASSERT
    assert out.shape == (batch_size, d_state), "Vmap output shape mismatch"
    assert not jnp.any(jnp.isnan(out)), "Output contains NaNs after vmap/jit"

def test_dissipative_friction_gradient_stability():
    """
    Test gradient stability to ensure that gradients only flow into the lower triangle
    and do not produce NaNs.
    """
    # ARRANGE
    d_state = 16
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(123))
    friction = DissipativeFriction(d_state=d_state, key=key_mod)
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    def loss_fn(mod, input_x):
        out = mod(input_x)
        return jnp.sum(out ** 2)

    # ACT
    loss, grads = eqx.filter_value_and_grad(loss_fn)(friction, x)
    
    # ASSERT
    assert not jnp.isnan(loss), "Loss is NaN"
    assert not jnp.any(jnp.isnan(grads.W)), "Gradients contain NaNs"
    
    # Extract strictly upper triangular parts of the gradient
    grad_upper = jnp.triu(grads.W, k=1)
    
    # Because we apply tril dynamically inside the property, 
    # the gradient for the upper triangle of the parameter W should be zero.
    assert jnp.allclose(grad_upper, 0.0), "Gradient leaked into the upper triangle of W"


def test_dissipative_friction_guardrails():
    """
    Test that the DissipativeFriction constructor correctly validates inputs
    and rejects garbage values.
    """
    # ARRANGE
    key = jax.random.PRNGKey(42)
    
    class MockGarbageHull:
        d_state = 16
        
    class MockMismatchedHull:
        d_state = 10
        d_internal = 2
        d_sensory = 2
        d_active = 2
        
    class MockValidHull:
        d_state = 16
        d_internal = 4
        d_sensory = 4
        d_active = 4
    
    # ACT / ASSERT
    with pytest.raises(ValueError, match="strictly positive integer"):
        DissipativeFriction(d_state=-5, key=key)
        
    with pytest.raises(ValueError, match="non-negative float"):
        DissipativeFriction(d_state=16, key=key, epsilon=-1.0)
        
    with pytest.raises(TypeError, match="must possess d_internal"):
        DissipativeFriction(d_state=16, key=key, hull=MockGarbageHull())
        
    with pytest.raises(ValueError, match="must exactly match d_state"):
        DissipativeFriction(d_state=16, key=key, hull=MockMismatchedHull())
        
    # Should not raise
    DissipativeFriction(d_state=16, key=key, hull=MockValidHull())


def test_dissipative_friction_topological_masking():
    """
    Test that supplying a MarkovHull correctly zeroes out the external-internal
    block of the Cholesky factor while maintaining strict positive semi-definiteness
    and gradient stability.
    """
    # ARRANGE
    d_state = 16
    d_internal = 4
    d_sensory = 4
    d_active = 4
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(999))
    epsilon = 1e-4
    
    class MockHull:
        def __init__(self):
            self.d_state = d_state
            self.d_internal = d_internal
            self.d_sensory = d_sensory
            self.d_active = d_active
            
    hull = MockHull()
    friction = DissipativeFriction(d_state=d_state, key=key_mod, epsilon=epsilon, hull=hull)
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    def loss_fn(mod, input_x):
        out = mod(input_x)
        return jnp.sum(out ** 2)
        
    # ACT
    L = friction.L
    gamma = friction.Gamma
    loss, grads = eqx.filter_value_and_grad(loss_fn)(friction, x)
    
    # ASSERT
    idx_s = d_internal
    idx_e = d_internal + d_sensory + d_active
    
    # Boundary Conditions & Shape Consistency
    # L_ie (external rows, internal cols) should be exactly 0
    L_ie = L[idx_e:, :idx_s]
    assert jnp.all(L_ie == 0.0), "External-Internal block of L was not zeroed out"
    
    # Invariant (PSD)
    eigenvalues, _ = jnp.linalg.eigh(gamma)
    assert jnp.all(eigenvalues >= epsilon - 1e-6), "Gamma lost PSD property due to masking"
    
    # Gradient Stability
    assert not jnp.isnan(loss), "Loss is NaN"
    assert not jnp.any(jnp.isnan(grads.W)), "Gradients contain NaNs due to masking"
