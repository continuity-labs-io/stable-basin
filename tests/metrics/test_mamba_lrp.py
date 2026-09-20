import torch
import pytest
from src.metrics.mamba_lrp import MambaLRPEpsilon
from src.harness.sensor_fusion_predictor import SensorFusionPredictor, SSMType
from src.core.substrate import get_optimal_device
import torch.nn as nn
import torch.nn.functional as F


def test_relevance_conservation_axiom():
    """
    Verifies that the MambaLRPEpsilon implementation mathematically conserves
    relevance from the output prediction all the way back to the input tensor.
    Sum(R_out) must approximately equal Sum(R_in).
    """
    device = get_optimal_device(allow_mps=False)  # CPU for deterministic math

    # Initialize a small test model
    model = SensorFusionPredictor(
        ssm_type=SSMType.MASR_MAMBA, modality_dims=[16], d_model=32, out_dim=16
    ).to(device)
    lrp = MambaLRPEpsilon(model, epsilon=1e-7)

    # Generate random biological tensor [Batch, Time, Channels]
    x = torch.rand(1, 20, 16).to(device)
    x = x + 0.1  # Ensure non-zero inputs
    mask = torch.ones(1, 20, 1).to(device)

    target_time_step = 15

    # Forward pass to calculate expected total output relevance
    preds, _, _ = model(x, mask)
    expected_relevance = preds[:, target_time_step, :].sum().item()

    # Backward pass LRP
    relevance_tensor = lrp.attribute(x, target_time_step, mask=mask)
    actual_relevance = relevance_tensor.sum().item()

    # Calculate the conservation error
    error = abs(expected_relevance - actual_relevance)

    # The LRP-epsilon rule should conserve relevance within a small epsilon bound
    assert error < 1e-2, (
        f"Relevance Violated! Expected: {expected_relevance}, Actual: {actual_relevance}"
    )


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
