import torch
import pytest
from src.models.attention.baseline_transformer import BaselineTransformer


def test_baseline_transformer_shapes():
    """Verify the transformer returns the expected shape."""
    model = BaselineTransformer(d_model=32, nhead=4, num_layers=2, max_len=100)

    # batch=2, seq_len=50, d_model=32
    latent_x = torch.randn(2, 50, 32)

    out = model(latent_x)
    assert out.shape == (2, 50, 32), f"Expected (2, 50, 32), got {out.shape}"


def test_baseline_transformer_strict_causality():
    """
    Verify the strict causal masking. A change at timestep T must have ZERO
    effect on any timestep < T.
    """
    model = BaselineTransformer(d_model=16, nhead=2, num_layers=2, max_len=100)
    model.eval()

    x1 = torch.randn(2, 50, 16)
    x2 = x1.clone()

    # Introduce a massive spike at t=25
    x2[:, 25, :] += torch.randn(2, 16) * 100.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Past (t < 25): Must be absolutely identical
    diff_past = (out1[:, :25, :] - out2[:, :25, :]).abs().max().item()
    assert diff_past < 1e-5, (
        f"Causality leak detected! Future influenced the past. Max diff: {diff_past}"
    )

    # Future (t >= 25): Must diverge due to the spike
    diff_future = (out1[:, 25:, :] - out2[:, 25:, :]).abs().mean().item()
    assert diff_future > 1e-5, "Future did not diverge after the spike."


def test_baseline_transformer_kwargs():
    """Verify that pos_embedding_scale and ff_expansion_factor configure the model properly."""
    d_model = 16
    scale = 0.5
    factor = 2
    
    model = BaselineTransformer(
        d_model=d_model,
        pos_embedding_scale=scale,
        ff_expansion_factor=factor
    )
    
    # Check pos_embedding scaling (std of standard normal * 0.5 should be approx 0.5)
    std = model.pos_embedding.std().item()
    assert abs(std - scale) < 0.1, f"Expected pos_embedding std ~ {scale}, got {std}"
    
    # Check feedforward expansion factor
    # In PyTorch, the first linear layer expands to dim_feedforward
    dim_ff = model.transformer.layers[0].linear1.out_features
    expected_dim_ff = d_model * factor
    assert dim_ff == expected_dim_ff, f"Expected dim_feedforward {expected_dim_ff}, got {dim_ff}"
    
    # Ensure forward pass runs smoothly with these args
    x = torch.randn(2, 20, 16)
    out = model(x)
    assert out.shape == (2, 20, 16)
