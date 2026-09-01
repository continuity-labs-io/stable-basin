import jax
import jax.numpy as jnp
from src.echo.physics.thermostat import Thermostat
from src.echo.architecture.observer import MarkovBlanketObserver

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

def test_forced_thermalizer_omega_seq():
    key = jax.random.PRNGKey(0)
    obs = MarkovBlanketObserver(
        d_internal=4,
        d_sensory=4,
        d_active=4,
        d_external=4,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=1,
        temperature=1.0,
        key=key
    )
    
    x_init = jnp.zeros(obs.hull.d_state)
    dt = 0.01
    
    seq_len = 5
    omega_seq = jnp.ones((seq_len, obs.hull.d_state)) * 2.0
    
    traj = obs.forced_unroll(key, x_init, dt, seq=None, omega_seq=omega_seq)
    
    assert traj.shape == (seq_len, obs.hull.d_state)
