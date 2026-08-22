import torch
import pytest
from src.icebox.metrics.mamba_lrp import MambaLRPEpsilon
from src.harness.sensor_fusion_predictor import SensorFusionPredictor, SSMType
from src.utils.device import get_optimal_device


def test_relevance_conservation_axiom():
    """
    Verifies that the MambaLRPEpsilon implementation mathematically conserves
    relevance from the output prediction all the way back to the input tensor.
    Sum(R_out) must approximately equal Sum(R_in).
    """
    device = get_optimal_device(allow_mps=False)  # CPU for deterministic math

    # Initialize a small test model
    model = SensorFusionPredictor(ssm_type=SSMType.MASR_MAMBA, modality_dims=[16], d_model=32, out_dim=16).to(device)
    lrp = MambaLRPEpsilon(model, epsilon=1e-7)

    # Generate random biological tensor [Batch, Time, Channels]
    x = torch.rand(1, 20, 16).to(device)
    x = x + 0.1  # Ensure non-zero inputs
    mask = torch.ones(1, 20, 1).to(device)

    target_time_step = 15

    # Forward pass to calculate expected total output relevance
    preds, _ = model(x, mask)
    expected_relevance = preds[:, target_time_step, :].sum().item()

    # Backward pass LRP
    relevance_tensor = lrp.attribute(x, target_time_step, mask=mask)
    actual_relevance = relevance_tensor.sum().item()

    # Calculate the conservation error
    error = abs(expected_relevance - actual_relevance)

    # The LRP-epsilon rule should conserve relevance within a small epsilon bound
    assert error < 1e-2, (
        f"Relevance Conservation Axiom Violated! Expected: {expected_relevance}, Actual: {actual_relevance}, Error: {error}"
    )
