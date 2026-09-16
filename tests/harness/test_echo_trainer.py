import jax
import jax.numpy as jnp
import pytest
import equinox as eqx

from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.harness.echo_trainer import EchoTrainer

def test_echo_trainer_step_bptt():
    """
    ARRANGE
    """
    key = jax.random.PRNGKey(42)
    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 2
    d_state = d_internal + d_sensory + d_active + d_external
    
    k_model, k_batch, k_step = jax.random.split(key, 3)
    
    model = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=8,
        ebm_depth=1,
        n_steps=5,
        temperature=1.0,
        key=k_model
    )
    
    trainer = EchoTrainer(model=model, learning_rate=1e-3)
    
    batch_size = 4
    seq_len = 5
    
    s_true = jax.random.normal(k_batch, (batch_size, seq_len, d_sensory))
    x_init = jax.random.normal(k_batch, (batch_size, d_state))
    
    batch = {
        's_true': s_true,
        'x_init': x_init
    }
    
    """
    ACT
    """
    new_model, new_trainer, loss = trainer.step(model, batch, k_step, dt=0.01)
    
    """
    ASSERT
    """
    assert jnp.isfinite(loss)
    
    # Check that model parameters have changed
    diff = jnp.sum((model.ebm.mlp.layers[0].weight - new_model.ebm.mlp.layers[0].weight) ** 2)
    assert diff > 0.0, "Model parameters should be updated by the trainer."

