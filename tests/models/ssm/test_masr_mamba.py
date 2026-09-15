import torch
import pytest
from src.models.ssm.masr_mamba import PyTorchMambaMASR, mamba_masr_reference_scan

def test_masr_mamba_output_shape():
    """Test that PyTorchMambaMASR returns the expected output shape."""
    batch, seq_len, d_model = 2, 10, 8
    d_state = 4
    
    model = PyTorchMambaMASR(d_model=d_model, d_state=d_state)
    
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.ones(batch, seq_len, d_model)
    
    y = model(x, mask)
    
    assert y.shape == (batch, seq_len, d_model), f"Expected shape {(batch, seq_len, d_model)}, got {y.shape}"

def test_mamba_masr_reference_scan_stasis():
    """Test that when mask=0, the hidden state perfectly freezes."""
    batch_size = 1
    seq_len = 5
    d_model = 2
    d_state = 1
    
    x = torch.ones(batch_size, seq_len, d_model)
    dt = torch.ones(batch_size, seq_len, d_model)
    
    # Create mask where time step 2 and 3 are missing for channel 0
    mask = torch.ones(batch_size, seq_len, d_model)
    mask[0, 2:4, 0] = 0.0
    
    A = torch.full((d_model, d_state), -0.5)
    B = torch.ones(batch_size, seq_len, d_state)
    # Set C to 1 so that the output y directly reflects h (if D=0)
    C = torch.ones(batch_size, seq_len, d_state)
    D = torch.zeros(d_model)
    
    y = mamba_masr_reference_scan(x, dt, mask, A, B, C, D)
    
    # For channel 0, since mask is 0 at t=2 and t=3, the state should freeze:
    # y[t=2] == y[t=1]
    # y[t=3] == y[t=2]
    
    assert torch.allclose(y[0, 2, 0], y[0, 1, 0]), "Hidden state did not freeze at t=2 when mask=0"
    assert torch.allclose(y[0, 3, 0], y[0, 2, 0]), "Hidden state did not freeze at t=3 when mask=0"
    
    # At t=4, mask is 1 again, so it should change
    assert not torch.allclose(y[0, 4, 0], y[0, 3, 0]), "Hidden state should update at t=4 when mask=1"
    
    # For channel 1, mask is always 1, so it should never freeze
    assert not torch.allclose(y[0, 2, 1], y[0, 1, 1]), "Channel 1 should not freeze, mask is 1"

def test_masr_mamba_gradients():
    """Test that backward pass works smoothly."""
    batch, seq_len, d_model, d_state = 2, 5, 4, 2
    model = PyTorchMambaMASR(d_model=d_model, d_state=d_state)
    
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    mask = torch.ones(batch, seq_len, d_model)
    
    y = model(x, mask)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients did not flow back to input x"
    assert model.A_init.A_log.grad is not None, "Gradients did not flow back to parameter A"
    assert model.B_proj.weight.grad is not None, "Gradients did not flow back to parameter B_proj"

def test_masr_mamba_device_compatibility():
    """Test that the model handles different devices correctly if available."""
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    d_model, d_state = 4, 2
    model = PyTorchMambaMASR(d_model=d_model, d_state=d_state).to(device)
    
    x = torch.randn(2, 5, d_model, device=device)
    mask = torch.ones(2, 5, d_model, device=device)
    
    y = model(x, mask)
    assert y.device.type == device.type, f"Expected output on {device.type}, got {y.device.type}"
