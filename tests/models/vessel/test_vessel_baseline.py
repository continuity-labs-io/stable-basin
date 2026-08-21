import torch
import pytest
from src.models.vessel.vessel_baseline import VesselBaseline

def test_vessel_baseline_masking_invariants():
    # ARRANGE
    size = 100
    model = VesselBaseline(size=size, dt=0.01, sigma_ext=1.0)
    # Enable gradients on states for stability test
    model.u.requires_grad = True
    model.v.requires_grad = True

    u_init = model.u
    v_init = model.v

    # ACT
    u_new, v_new, internal_var = model()
    
    # We will compute a loss that depends on u_new, v_new and internal_var and backpropagate
    loss = u_new.sum() + v_new.sum() + internal_var
    loss.backward()

    # ASSERT
    # Assertion 1: Shape consistency
    assert u_new.shape == (1, 1, size, size), "Shape consistency failed for u_new"
    assert v_new.shape == (1, 1, size, size), "Shape consistency failed for v_new"
    assert internal_var.ndim == 0, "Shape consistency failed for internal_var (should be scalar)"
    
    # Assertion 2: Boundary conditions and Mask constraints
    # Check that v is rigidly maintained at threshold in the wall
    v_wall_post = v_new[model.mask_wall]
    assert torch.allclose(v_wall_post, torch.tensor(model.v_wall_threshold, dtype=v_new.dtype, device=v_new.device)), \
        "Boundary condition failed: v is not rigidly maintained at threshold in the wall"
    
    # Check that D_u and D_v are reduced in the wall to dampen noise
    expected_D_u_wall = torch.tensor(0.16 * 0.01, dtype=model.D_u.dtype)
    assert torch.allclose(model.D_u[model.mask_wall], expected_D_u_wall), "Mask constraint failed for D_u wall"
    
    # Assertion 3: Gradient stability
    assert u_init.grad is not None, "Gradient stability failed: missing grad on u"
    assert v_init.grad is not None, "Gradient stability failed: missing grad on v"
    assert not torch.isnan(u_init.grad).any(), "Gradient stability failed: NaNs in u gradient"
    assert not torch.isnan(v_init.grad).any(), "Gradient stability failed: NaNs in v gradient"
    assert not torch.isinf(u_init.grad).any(), "Gradient stability failed: Infs in u gradient"
