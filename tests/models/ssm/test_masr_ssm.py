import torch
import pytest
from src.models.ssm.masr_ssm import MaskAwareSSM
from src.models.ssm.baseline_ssm import BaselineSSM

def test_mask_aware_ssm_shape():
    batch, seq_len, d_model = 2, 10, 16
    model = MaskAwareSSM(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model)
    g = torch.ones(batch, seq_len, d_model)
    out = model(x, g)
    assert out.shape == (batch, seq_len, d_model), "Output shape should match input shape"

def test_mask_aware_ssm_time_freezing():
    batch, seq_len, d_model = 2, 5, 8
    model = MaskAwareSSM(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model)
    g = torch.ones(batch, seq_len, d_model)
    
    # Freeze the third timestep (index 2) for all batches
    g[:, 2, :] = 0.0
    
    out = model(x, g)
    
    # Check that output at t=2 is perfectly identical to output at t=1
    # Because when g_t = 0, dt=0 -> A_bar = 1, B_bar = 0 -> h(t) = h(t-1)
    assert torch.allclose(out[:, 2, :], out[:, 1, :]), "State was not frozen when gate was 0"
    
    # Ensure it's not identically 0 elsewhere
    assert not torch.allclose(out[:, 1, :], out[:, 0, :]), "States should normally change"

def test_mask_aware_ssm_equivalence_to_baseline():
    batch, seq_len, d_model = 2, 5, 8
    
    # Create both models
    mask_model = MaskAwareSSM(d_model=d_model)
    base_model = BaselineSSM(d_model=d_model)
    
    # Sync weights exactly
    base_model.load_state_dict(mask_model.state_dict())
    
    x = torch.randn(batch, seq_len, d_model)
    g = torch.ones(batch, seq_len, d_model)
    
    out_mask = mask_model(x, g)
    out_base = base_model(x)
    
    # They should be extremely close (the only difference is the 1e-8 shift in dt_gated in the mask model)
    assert torch.allclose(out_mask, out_base, atol=1e-5), "MaskAwareSSM with g=1 should match BaselineSSM"

def test_mask_aware_ssm_backward_pass():
    batch, seq_len, d_model = 2, 5, 8
    model = MaskAwareSSM(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    g = torch.ones(batch, seq_len, d_model, requires_grad=True)
    
    out = model(x, g)
    loss = out.sum()
    loss.backward()
    
    # Check that gradients flow to both inputs and all parameters
    assert x.grad is not None
    assert g.grad is not None
    assert model.A_init.A_log.grad is not None
    assert model.B_proj.weight.grad is not None
    assert model.dt_proj.weight.grad is not None
