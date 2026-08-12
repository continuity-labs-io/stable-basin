import torch
import pytest
from src.models.encoders.orthogonal_modality_encoder import OrthogonalModalityEncoder

def test_proportional_orthogonal_routing_edge_case():
    """
    Test edge case where one modality heavily outweighs the other.
    With modality_dims=[1000, 2] and d_model=64:
    Total dim = 1002.
    Modality 0 proportion = 1000 / 1002 = ~0.998
    Chunk size 0 = int(64 * 0.998) = 63.
    Modality 1 gets the remaining 1 dimension.
    """
    modality_dims = [1000, 2]
    d_model = 64
    d_in = sum(modality_dims)

    encoder = OrthogonalModalityEncoder(d_in=d_in, modality_dims=modality_dims, d_model=d_model)

    w_gate_weight = encoder.W_gate.weight
    
    # Modality 0 should have 20.0 for first 63 dims
    assert torch.all(w_gate_weight[:63, 0] == 20.0)
    # Modality 0 should be 0.0 for the rest
    assert torch.all(w_gate_weight[63:, 0] == 0.0)

    # Modality 1 should be 0.0 for first 63 dims
    assert torch.all(w_gate_weight[:63, 1] == 0.0)
    # Modality 1 should have 20.0 for the last dim
    assert torch.all(w_gate_weight[63:, 1] == 20.0)

def test_proportional_orthogonal_routing_equal():
    """
    Test standard equal division.
    """
    modality_dims = [20, 20]
    d_model = 64
    d_in = 40

    encoder = OrthogonalModalityEncoder(d_in=d_in, modality_dims=modality_dims, d_model=d_model)

    w_gate_weight = encoder.W_gate.weight
    
    # Should be 32/32 split
    assert torch.all(w_gate_weight[:32, 0] == 20.0)
    assert torch.all(w_gate_weight[32:, 0] == 0.0)
    
    assert torch.all(w_gate_weight[:32, 1] == 0.0)
    assert torch.all(w_gate_weight[32:, 1] == 20.0)

def test_too_many_modalities_error():
    # If d_model < n_modalities, should raise ValueError
    with pytest.raises(ValueError):
        OrthogonalModalityEncoder(d_in=10, modality_dims=[2, 2, 2], d_model=2)
