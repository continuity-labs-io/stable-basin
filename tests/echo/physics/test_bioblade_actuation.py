import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from src.echo.physics.thermostat import Thermostat
from src.echo.architecture.observer import MarkovBlanketObserver

def test_bioblade_actuation_thermostat():
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
    drift = -(Q - Gamma) @ grad_E
    
    expected_x_next = x + (drift * dt) + (q_ext * dt)
    
    assert jnp.allclose(x_next, expected_x_next)


def test_bioblade_actuation_observer():
    key = jax.random.PRNGKey(42)
    obs_key, sim_key = jax.random.split(key, 2)
    
    observer = MarkovBlanketObserver(
        d_internal=2,
        d_sensory=2,
        d_active=2,
        d_external=2,
        ebm_hidden_size=16,
        ebm_depth=1,
        n_steps=1,
        temperature=0.0,
        key=obs_key
    )
    
    seq_len = 10
    d_state = 8
    dt = 0.1
    x_init = jnp.zeros(d_state)
    
    q_seq = jnp.ones((seq_len, d_state)) * 2.0
    
    # Assert it executes without JAX concretization or shape errors
    traj = observer.forced_unroll(sim_key, x_init, dt, seq=None, omega_seq=None, q_seq=q_seq)
    
    assert traj.shape == (seq_len, d_state)
    assert not jnp.any(jnp.isnan(traj))
