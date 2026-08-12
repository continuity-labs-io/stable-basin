import torch
import pytest
from src.models.ssm.baseline_ssm import BaselineSSM

def test_baseline_ssm_shape():
    batch, seq_len, d_model = 2, 10, 16
    model = BaselineSSM(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model)
    out = model(x)
    assert out.shape == (batch, seq_len, d_model), "Output shape should match input shape"

def test_baseline_ssm_batch_independence():
    batch, seq_len, d_model = 2, 5, 8
    model = BaselineSSM(d_model=d_model)
    
    # Create a batch with two identical sequences
    x_single = torch.randn(1, seq_len, d_model)
    x = x_single.expand(batch, -1, -1).clone()
    
    # Modify only the first sequence
    x[0, -1, :] += 10.0
    
    out = model(x)
    
    # Run a purely single batch to compare
    out_single = model(x_single)
    
    # The output for the second sequence in the batch should perfectly match the single batch
    assert torch.allclose(out[1], out_single[0], atol=1e-6), "Batches are not independent!"

def test_baseline_ssm_initialization_range():
    d_model = 16
    model = BaselineSSM(d_model=d_model)
    A = -torch.exp(model.A_log).detach()
    
    # A should be in [-0.6, -0.1]
    # We allow a tiny bit of float imprecision just in case
    assert torch.all(A <= -0.1 + 1e-6), "A contains values > -0.1"
    assert torch.all(A >= -0.6 - 1e-6), "A contains values < -0.6"

def test_baseline_ssm_stability_bounds():
    batch, seq_len, d_model = 2, 5, 8
    model = BaselineSSM(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model)
    
    A = -torch.exp(model.A_log)
    
    # Manually check A_bar computation on all x
    dt = torch.nn.functional.softplus(model.dt_proj(x))
    
    # A_bar should be in (0, 1) because A < 0 and dt > 0
    # A is (d_model), dt is (batch, seq_len, d_model)
    A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt) 
    
    assert torch.all(A_bar > 0), "A_bar has values <= 0"
    assert torch.all(A_bar < 1.0), "A_bar has values >= 1.0"
    
def test_baseline_ssm_no_nan_inf():
    d_model = 16
    model = BaselineSSM(d_model=d_model)
    
    # Edge case: All zeros
    x_zeros = torch.zeros(2, 5, d_model)
    out_zeros = model(x_zeros)
    assert not torch.isnan(out_zeros).any(), "NaN in output with zero input"
    assert not torch.isinf(out_zeros).any(), "Inf in output with zero input"
    
    # Edge case: Large values
    x_large = torch.randn(2, 5, d_model) * 1000
    out_large = model(x_large)
    assert not torch.isnan(out_large).any(), "NaN in output with large input"
    assert not torch.isinf(out_large).any(), "Inf in output with large input"

def test_baseline_ssm_backward_pass():
    batch, seq_len, d_model = 2, 5, 8
    model = BaselineSSM(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # Check that gradients are populated
    assert x.grad is not None, "Gradient did not flow back to input x"
    assert model.A_log.grad is not None, "Gradient did not flow to A_log"
    assert model.B_proj.weight.grad is not None, "Gradient did not flow to B_proj"
    assert model.dt_proj.weight.grad is not None, "Gradient did not flow to dt_proj"
    
    # Ensure gradients are not all zeros
    assert not torch.allclose(model.A_log.grad, torch.zeros_like(model.A_log.grad)), "Gradients for A_log are exactly zero"
