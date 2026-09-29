import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from src.models.encoders.spatial_compressor import SpatialCompressor


class DummyViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Just a dummy projection to satisfy the shape [Batch, 768]
        # and allow gradients to flow back to the input
        self.proj = nn.Linear(3 * 224 * 224, 768)

    def forward(self, x):
        return self.proj(x.reshape(x.shape[0], -1))


def dummy_create_model(*args, **kwargs):
    return DummyViT()


# torch.autograd.set_detect_anomaly(True)

@patch("src.models.encoders.spatial_compressor.timm.create_model", side_effect=dummy_create_model)
def test_spatial_compressor_backward_gradient_flow(mock_create):
    # ARRANGE
    B, T, C, D, H, W = 1, 2, 2, 4, 128, 128
    device = "cpu"
    x = torch.randn(B, T, C, D, H, W, device=device, requires_grad=True)
    
    # Initialize the model and ensure it's on the correct device
    model = SpatialCompressor(model_name="vit_base_patch16_224").to(device)
    
    # ACT
    # Pass the tensor through the model
    out = model(x)
    
    # Simulate a backward pass (e.g., from a downstream loss)
    loss = out.sum()
    loss.backward()
    
    # ASSERT
    # Gradients should flow backward to the input tensor `x`
    assert x.grad is not None, "Gradient did not flow back to input x. x.grad is None."
    assert not torch.isnan(x.grad).any(), "Gradient contains NaNs."
    assert x.grad.shape == x.shape, "Gradient shape mismatch."


@patch("src.models.encoders.spatial_compressor.timm.create_model", side_effect=dummy_create_model)
def test_spatial_compressor_extreme_values(mock_create):
    # ARRANGE
    B, T, C, D, H, W = 2, 1, 2, 2, 224, 224
    device = "cpu"
    # Extremely large and small values
    x = torch.tensor([1e8, -1e8, 0.0, 1e-8], device=device).view(1, 1, 1, 1, 2, 2)
    x = x.repeat(B, T, C, D, 112, 112)
    x.requires_grad = True
    
    model = SpatialCompressor(model_name="vit_base_patch16_224").to(device)
    
    # ACT
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # ASSERT
    assert not torch.isnan(out).any(), "Forward pass returned NaNs with extreme inputs."
    if x.grad is not None:
        assert not torch.isnan(x.grad).any(), "Backward pass returned NaNs with extreme inputs."

