import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from src.echo.physics.thermostat import Thermostat


def test_thermostat_shape_and_nans():
    """
    Test that the output shape matches input and contains no NaNs.
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    temperature = 1.0
    key_x, key_grad, key_w1, key_w2, key_noise = jax.random.split(jax.random.PRNGKey(42), 5)

    thermostat = Thermostat(temperature=temperature)

    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    grad_E = jax.random.normal(key_grad, (d_state,), dtype=jnp.float32)

    # Q must be skew-symmetric
    W1 = jax.random.normal(key_w1, (d_state, d_state), dtype=jnp.float32)
    Q = W1 - W1.T

    # L must be lower triangular
    W2 = jax.random.normal(key_w2, (d_state, d_state), dtype=jnp.float32)
    L = jnp.tril(W2)

    # ACT
    x_next = thermostat(x, grad_E, Q, L, dt, key_noise)

    # ASSERT
    assert x_next.shape == x.shape, "Output shape does not match input shape"
    assert not jnp.any(jnp.isnan(x_next)), "Output contains NaNs"


def test_thermostat_zero_temperature():
    """
    Test that at zero temperature, the output exactly equals the deterministic
    Euler step.
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    temperature = 0.0
    key_x, key_grad, key_w1, key_w2, key_noise = jax.random.split(jax.random.PRNGKey(123), 5)

    thermostat = Thermostat(temperature=temperature)

    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    grad_E = jax.random.normal(key_grad, (d_state,), dtype=jnp.float32)

    W1 = jax.random.normal(key_w1, (d_state, d_state), dtype=jnp.float32)
    Q = W1 - W1.T

    W2 = jax.random.normal(key_w2, (d_state, d_state), dtype=jnp.float32)
    L = jnp.tril(W2)

    # ACT
    x_next = thermostat(x, grad_E, Q, L, dt, key_noise)

    expected_drift = -(Q + (L @ L.T)) @ grad_E
    x_next_expected = x + (expected_drift * dt)

    # ASSERT
    assert jnp.allclose(x_next, x_next_expected, atol=1e-5), (
        "Zero temperature update does not match deterministic drift"
    )


def test_thermostat_jit():
    """
    Test that the module can be successfully JIT-compiled.
    """
    # ARRANGE
    d_state = 4
    dt = 0.1
    key_x, key_grad, key_w1, key_w2, key_noise = jax.random.split(jax.random.PRNGKey(99), 5)

    thermostat = Thermostat(temperature=1.0)

    x = jax.random.normal(key_x, (d_state,), dtype=jnp.float32)
    grad_E = jax.random.normal(key_grad, (d_state,), dtype=jnp.float32)

    W1 = jax.random.normal(key_w1, (d_state, d_state), dtype=jnp.float32)
    Q = W1 - W1.T

    W2 = jax.random.normal(key_w2, (d_state, d_state), dtype=jnp.float32)
    L = jnp.tril(W2)

    # ACT
    @eqx.filter_jit
    def jit_forward(mod, *args):
        return mod(*args)

    x_next = jit_forward(thermostat, x, grad_E, Q, L, dt, key_noise)

    # ASSERT
    assert x_next.shape == (d_state,)
    assert not jnp.any(jnp.isnan(x_next))


def test_thermostat_vmap():
    """
    Test that the module can be batched across x, grad_E, and keys, while
    keeping Q, L, and dt shared across the batch.
    """
    # ARRANGE
    batch_size = 16
    d_state = 4
    dt = 0.1
    key_x, key_grad, key_w1, key_w2, key_noise = jax.random.split(jax.random.PRNGKey(4242), 5)

    thermostat = Thermostat(temperature=1.0)

    # Batched inputs
    x_batch = jax.random.normal(key_x, (batch_size, d_state), dtype=jnp.float32)
    grad_E_batch = jax.random.normal(key_grad, (batch_size, d_state), dtype=jnp.float32)
    keys_batch = jax.random.split(key_noise, batch_size)

    # Shared inputs
    W1 = jax.random.normal(key_w1, (d_state, d_state), dtype=jnp.float32)
    Q = W1 - W1.T

    W2 = jax.random.normal(key_w2, (d_state, d_state), dtype=jnp.float32)
    L = jnp.tril(W2)

    # ACT
    # vmap signature: thermostat(x, grad_E, Q, L, dt, key)
    # x: batched (0)
    # grad_E: batched (0)
    # Q: shared (None)
    # L: shared (None)
    # dt: shared (None)
    # key: batched (0)
    batched_forward = eqx.filter_vmap(thermostat, in_axes=(0, 0, None, None, None, 0))

    x_next_batch = batched_forward(x_batch, grad_E_batch, Q, L, dt, keys_batch)

    # ASSERT
    assert x_next_batch.shape == (batch_size, d_state), "Batched output shape mismatch"
    assert not jnp.any(jnp.isnan(x_next_batch)), "Batched output contains NaNs"


def test_actuation_thermostat():
    # Instantiate a Thermostat with temperature = 0.0.
    thermostat = Thermostat(temperature=0.0)

    d_state = 2
    x = jnp.array([1.0, -1.0])
    grad_E = jnp.array([0.1, -0.1])
    Q = jnp.zeros((d_state, d_state))
    L = jnp.eye(d_state)
    dt = 0.1
    key = jax.random.PRNGKey(0)

    q_ext = jnp.array([5.0, -5.0])

    x_next = thermostat(x, grad_E, Q, L, dt, key, q_ext=q_ext)

    Gamma = L @ L.T
    drift = -(Q + Gamma) @ grad_E

    expected_x_next = x + (drift * dt) + (q_ext * dt)

    assert jnp.allclose(x_next, expected_x_next)


def test_thermostat_omega_ext():
    thermostat = Thermostat(temperature=0.0)
    d_state = 10
    x = jnp.zeros(d_state)
    grad_E = jnp.zeros(d_state)
    Q = jnp.zeros((d_state, d_state))
    L = jnp.zeros((d_state, d_state))
    dt = 0.01
    key = jax.random.PRNGKey(0)

    x_next_no_force = thermostat(x, grad_E, Q, L, dt, key, omega_ext=None)

    omega_ext = jnp.ones(d_state) * 10.0
    x_next_force = thermostat(x, grad_E, Q, L, dt, key, omega_ext=omega_ext)

    shift = x_next_force - x_next_no_force
    expected_shift = omega_ext * dt

    assert jnp.allclose(shift, expected_shift)
