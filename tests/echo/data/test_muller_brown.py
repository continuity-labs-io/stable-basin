import pytest
import jax
import jax.numpy as jnp
import torch

from src.echo.data.toy.muller_brown import MullerBrownDataset, muller_brown_potential, generate_trajectory

def test_muller_brown_dataset_shape_and_nans():
    """
    Shape & Interoperability Test:
    Instantiates dataset and asserts correct lengths, shapes, and lack of NaNs.
    """
    size = 10
    seq_len = 100
    dataset = MullerBrownDataset(size=size, seq_len=seq_len)
    
    # Assert __len__
    assert len(dataset) == size
    
    # Fetch first sequence
    sample = dataset[0]
    
    # Assert dictionary keys
    assert "x_raw" in sample
    assert "mask" in sample
    assert "y_true" in sample
    
    # Assert PyTorch tensors and shapes
    x_raw = sample["x_raw"]
    assert isinstance(x_raw, torch.Tensor)
    assert x_raw.shape == (seq_len, 2)
    
    # Assert identical shapes
    assert sample["mask"].shape == (seq_len, 2)
    assert sample["y_true"].shape == (seq_len, 2)
    
    # Assert y_true matches x_raw
    assert torch.allclose(x_raw, sample["y_true"])
    
    # Assert no NaNs
    assert not torch.isnan(x_raw).any()


def test_muller_brown_minimum():
    """
    The Minimum Test:
    Evaluates the analytical gradient at the approximate global minimum.
    Asserts gradient magnitude is near zero.
    """
    approx_min = jnp.array([-0.558, 1.441])
    
    # Compute analytical gradient
    grad = jax.grad(muller_brown_potential)(approx_min)
    
    # Compute magnitude (L2 norm)
    grad_magnitude = jnp.linalg.norm(grad)
    
    # Assert near zero
    assert grad_magnitude < 3.0, f"Gradient magnitude {grad_magnitude} at minimum is too large."


def test_muller_brown_fast_jax_execution():
    """
    Fast JAX Execution Test:
    Asserts generate_trajectory can be executed efficiently under jax.jit.
    """
    key = jax.random.PRNGKey(99)
    state_init = jnp.array([-0.5, 1.5])
    n_steps = 10
    dt = 0.001
    kT = 15.0
    
    # JIT compilation is already applied via decorator in module,
    # but we can explicitly test that calling it doesn't raise concretization errors.
    try:
        trajectory = generate_trajectory(key, state_init, n_steps, dt, kT)
    except Exception as e:
        pytest.fail(f"JIT execution failed with error: {e}")
        
    assert trajectory.shape == (n_steps, 2)
    assert not jnp.any(jnp.isnan(trajectory))
