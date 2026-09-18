import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from src.echo.physics.solenoidal import SolenoidalFlow

def test_solenoidal_flow_skew_symmetry():
    """Asserts that the computed matrix Q is perfectly skew-symmetric."""
    d_state = 64
    key = jax.random.PRNGKey(42)
    model = SolenoidalFlow(d_state=d_state, key=key)
    
    Q = model.Q
    
    # Q should equal -Q.T
    assert jnp.allclose(Q, -Q.T, atol=1e-6)

def test_solenoidal_flow_thermodynamic_invariant():
    """Asserts that x^T Q x = 0 (no thermodynamic work)."""
    d_state = 64
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    model = SolenoidalFlow(d_state=d_state, key=key1)
    
    # Random state vector
    x = jax.random.normal(key2, (d_state,))
    
    # Qx
    Q_x = model(x)
    
    # x^T Q x
    work = jnp.dot(x, Q_x)
    
    assert jnp.abs(work) < 1e-5

def test_solenoidal_flow_jit():
    """Asserts that the module can be successfully JIT-compiled."""
    d_state = 64
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    model = SolenoidalFlow(d_state=d_state, key=key1)
    x = jax.random.normal(key2, (d_state,))
    
    @jax.jit
    def apply_model(m, inp):
        return m(inp)
        
    # First call will trace and compile
    out_jit = apply_model(model, x)
    out_eager = model(x)
    
    assert jnp.allclose(out_jit, out_eager, atol=1e-6)

def test_solenoidal_flow_vmap():
    """Asserts that the module can be batched over a dimension using jax.vmap."""
    d_state = 64
    batch_size = 16
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    
    model = SolenoidalFlow(d_state=d_state, key=key1)
    
    # Batch of random state vectors
    x_batch = jax.random.normal(key2, (batch_size, d_state))
    
    # vmap the module over the input batch
    # in_axes=(None, 0) because model is passed as first argument, x as second
    apply_vmap = jax.vmap(lambda m, inp: m(inp), in_axes=(None, 0))
    
    out_vmap = apply_vmap(model, x_batch)
    
    assert out_vmap.shape == (batch_size, d_state)
    
    # Check that it matches a sequential apply
    out_seq = jnp.stack([model(x) for x in x_batch])
    assert jnp.allclose(out_vmap, out_seq, atol=1e-6)


def test_solenoidal_flow_guardrails():
    """
    Test that the SolenoidalFlow constructor correctly validates inputs
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
        SolenoidalFlow(d_state=-5, key=key)
        
    with pytest.raises(TypeError, match="must possess d_internal"):
        SolenoidalFlow(d_state=16, key=key, hull=MockGarbageHull())
        
    with pytest.raises(ValueError, match="must exactly match d_state"):
        SolenoidalFlow(d_state=16, key=key, hull=MockMismatchedHull())
        
    # Should not raise
    SolenoidalFlow(d_state=16, key=key, hull=MockValidHull())


def test_solenoidal_flow_topological_masking():
    """
    Test that supplying a MarkovHull correctly applies topological masking to Q,
    while maintaining skew-symmetry and thermodynamic invariants (work = 0).
    """
    # ARRANGE
    d_state = 16
    d_internal = 4
    d_sensory = 4
    d_active = 4
    key_mod, key_x = jax.random.split(jax.random.PRNGKey(999))
    
    class MockHull:
        def __init__(self):
            self.d_state = d_state
            self.d_internal = d_internal
            self.d_sensory = d_sensory
            self.d_active = d_active
            
        def get_topology_mask(self):
            mask = jnp.ones((self.d_state, self.d_state), dtype=jnp.float32)
            idx_s = self.d_internal
            idx_e = self.d_internal + self.d_sensory + self.d_active
            mask = mask.at[idx_e:, :idx_s].set(0.0)
            mask = mask.at[:idx_s, idx_e:].set(0.0)
            return mask
            
    hull = MockHull()
    model = SolenoidalFlow(d_state=d_state, key=key_mod, hull=hull)
    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    
    def loss_fn(mod, input_x):
        out = mod(input_x)
        return jnp.sum(out ** 2)
        
    # ACT
    Q = model.Q
    Q_x = model(x)
    work = jnp.dot(x, Q_x)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    
    # ASSERT
    idx_s = d_internal
    idx_e = d_internal + d_sensory + d_active
    
    # Boundary Conditions & Shape Consistency
    # Q_ie (external rows, internal cols) and Q_ei should be exactly 0
    Q_ie = Q[idx_e:, :idx_s]
    Q_ei = Q[:idx_s, idx_e:]
    assert jnp.all(Q_ie == 0.0), "External-Internal block of Q was not zeroed out"
    assert jnp.all(Q_ei == 0.0), "Internal-External block of Q was not zeroed out"
    
    # Invariants
    # Skew-symmetric
    assert jnp.allclose(Q, -Q.T, atol=1e-6), "Q lost skew-symmetry due to masking"
    # No thermodynamic work
    assert jnp.abs(work) < 1e-5, "Thermodynamic work is not zero due to masking"
    
    # Gradient Stability
    assert not jnp.isnan(loss), "Loss is NaN"
    assert not jnp.any(jnp.isnan(grads.W)), "Gradients contain NaNs due to masking"
