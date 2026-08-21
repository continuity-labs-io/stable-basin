import torch
import pytest
from src.models.vessel.observer_zero import ObserverZero

def test_laplacian_invariants():
    # ARRANGE
    size = 16
    model = ObserverZero(size=size)
    x = torch.zeros(1, 1, size, size, requires_grad=True)
    # Put a pulse at the origin (0, 0)
    with torch.no_grad():
        x[0, 0, 0, 0] = 1.0

    # ACT
    out = model._laplacian(x)
    loss = out.sum()
    loss.backward()

    # ASSERT
    # Assertion 1: Shape consistency
    assert out.shape == x.shape, "Shape consistency failed for Laplacian"
    
    # Assertion 2: Boundary conditions (Toroidal wrap-around)
    # The kernel is 3x3, centered. The corners of the pulse at (0, 0) should wrap to (15, 15), (0, 15), (15, 0)
    assert torch.isclose(out[0, 0, size-1, size-1], torch.tensor(0.05)), "Boundary condition failed: diagonal wrap"
    assert torch.isclose(out[0, 0, size-1, 0], torch.tensor(0.20)), "Boundary condition failed: vertical wrap"
    assert torch.isclose(out[0, 0, 0, size-1], torch.tensor(0.20)), "Boundary condition failed: horizontal wrap"
    
    # Assertion 3: Gradient stability
    assert x.grad is not None, "Gradient stability failed: no gradient"
    assert not torch.isnan(x.grad).any(), "Gradient stability failed: NaNs in gradient"
    assert not torch.isinf(x.grad).any(), "Gradient stability failed: Infs in gradient"


def test_continuous_update_invariants():
    # ARRANGE
    size = 16
    model = ObserverZero(size=size, dt=0.1, sigma=0.0) # sigma=0 for deterministic gradient check
    # Require grad on buffers for testing
    model.u.requires_grad = True
    model.v.requires_grad = True
    
    # Keep references to the leaf tensors
    u_init = model.u
    v_init = model.v

    # ACT
    u_new, v_new = model()
    loss = u_new.mean() + v_new.mean()
    loss.backward()

    # ASSERT
    # Assertion 1: Shape consistency
    assert u_new.shape == (1, 1, size, size), "Shape consistency failed for u_new"
    assert v_new.shape == (1, 1, size, size), "Shape consistency failed for v_new"
    
    # Assertion 2: Boundary conditions (State integration bounds)
    # Check that update is bounded and smooth, and values are physically plausible 
    # For a small dt, max change should be bounded.
    max_change_u = torch.max(torch.abs(u_new - u_init))
    assert max_change_u < 10.0, "Boundary condition failed: update step exploded"
    
    # Assertion 3: Gradient stability
    assert u_init.grad is not None, "Gradient stability failed: u missing grad"
    assert v_init.grad is not None, "Gradient stability failed: v missing grad"
    assert not torch.isnan(u_init.grad).any(), "Gradient stability failed: NaNs in u.grad"
    assert not torch.isnan(v_init.grad).any(), "Gradient stability failed: NaNs in v.grad"
