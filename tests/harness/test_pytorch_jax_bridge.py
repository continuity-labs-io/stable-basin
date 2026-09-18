import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from src.harness.pytorch_jax_bridge import torch_to_jax

def test_torch_to_jax_conversion():
    """
    Verifies that a PyTorch tensor is correctly converted to a JAX array
    using zero-copy DLPack bridge.
    """
    # ARRANGE
    shape = (4, 4)
    # create on CPU for general testing
    torch_tensor = torch.randn(shape, device='cpu')
    expected_values = torch_tensor.numpy()
    
    # ACT
    jax_array = torch_to_jax(torch_tensor)
    
    # ASSERT
    assert isinstance(jax_array, jax.Array), "Output should be a JAX Array"
    assert jax_array.shape == shape, "Shape should match"
    np.testing.assert_allclose(np.array(jax_array), expected_values, err_msg="Values should match exactly")

def test_torch_to_jax_type_error():
    """
    Verifies that passing a non-tensor raises a TypeError.
    """
    # ARRANGE
    invalid_input = [1, 2, 3]
    
    # ACT & ASSERT
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        torch_to_jax(invalid_input)
