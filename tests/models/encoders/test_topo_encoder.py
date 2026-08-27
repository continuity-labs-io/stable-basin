import pytest
import torch

from src.models.encoders.topo_encoder import TopoEncoder
from src.models.ssm.baseline_ssm import BaselineSSM

def test_topo_encoder():
    ssm = BaselineSSM(d_model=64, d_state=16)
    model = TopoEncoder(ssm=ssm, d_model=64)

    batch = 2
    time = 5
    # Input shape: [Batch, Time, 2, 64, 64]
    x = torch.randn(batch, time, 2, 64, 64)

    # Test without hidden states
    out = model(x, return_hidden=False)
    assert out.shape == (batch, 64)

    # Test with hidden states
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (batch, 64)
    assert hidden.shape == (batch, time, 64)
