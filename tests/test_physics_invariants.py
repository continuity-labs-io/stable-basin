import pytest
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.echo.physics.solenoidal import SolenoidalFlow
from src.icebox.metrics.mamba_lrp import MambaLRPEpsilon

def test_solenoidal_flow_antisymmetry():
    d_state = 16
    key = jax.random.PRNGKey(42)
    flow = SolenoidalFlow(d_state=d_state, key=key)
    
    Q = flow.Q
    
    # Assert Q is skew-symmetric
    assert jnp.allclose(Q, -Q.T, atol=1e-6)
    
    # Assert v^T Q v = 0
    v = jax.random.normal(jax.random.PRNGKey(99), (d_state,))
    quadratic_form = v.T @ Q @ v
    
    assert jnp.allclose(quadratic_form, 0.0, atol=1e-6)


class DummyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_proj = nn.Linear(5, 5)
        self.W_gate = nn.Linear(5, 5)  # Required by attribute mask initialization

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = DummyFusion()
        self.readout = nn.Linear(5, 5)
        
    def get_hidden_states(self, x, mask=None):
        # Dummy return of hidden states
        return x

    def forward(self, x):
        h = self.get_hidden_states(x)
        return self.readout(h)

def test_mamba_lrp_relevance_conservation():
    model = DummyModel()
    model.eval()
    
    lrp = MambaLRPEpsilon(model=model)
    
    x = torch.randn(1, 10, 5)
    
    # Run attribution
    target_time_step = 9
    R_x = lrp.attribute(x, target_time_step=target_time_step)
    
    # Reconstruct what the attribute method calculated as 'preds'
    hidden_states = model.get_hidden_states(x)
    W_out = model.readout.weight.data
    b_out = model.readout.bias.data
    preds = F.linear(hidden_states, W_out, b_out)
    
    total_relevance = R_x.sum().item()
    total_prediction = preds[:, target_time_step, :].sum().item()
    
    # Assert relevance is conserved within 1% relative tolerance
    assert torch.isclose(torch.tensor(total_relevance), torch.tensor(total_prediction), rtol=0.01)
