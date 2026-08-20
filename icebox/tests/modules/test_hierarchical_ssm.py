import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/modules')))
from hierarchical_ssm import HierarchicalSSM


def test_hierarchical_ssm():
    # Use lightweight dims
    model = HierarchicalSSM(d1=16, d2=4, dt=0.01, tau_delay_steps=10)

    steps = 50
    u = torch.randn(steps)
    K = 1.0

    x1_hist, x2_hist = model(u, K, steps)

    assert x1_hist.shape == (steps, 16)
    assert x2_hist.shape == (steps, 4)

    # Assert values are valid (no NaNs)
    assert not torch.isnan(x1_hist).any()
    assert not torch.isnan(x2_hist).any()
