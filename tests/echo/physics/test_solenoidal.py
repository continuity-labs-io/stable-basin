import jax
import jax.numpy as jnp
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
