import torch
import pytest
from src.models.vessel.active_inference_agent import ActiveInferenceAgent

def test_gradient_operator_invariants():
    # ARRANGE
    size = 16
    model = ActiveInferenceAgent(size=size)
    x = torch.zeros(1, 1, size, size, requires_grad=True)
    # Put a linear slope across the x-axis to test the x-gradient
    with torch.no_grad():
        for col in range(size):
            x[0, 0, :, col] = float(col)

    # ACT
    grad_x, grad_y = model.compute_gradient(x)
    loss = grad_x.sum() + grad_y.sum()
    loss.backward()

    # ASSERT
    # Assertion 1: Shape consistency
    assert grad_x.shape == x.shape, "Shape consistency failed for grad_x"
    assert grad_y.shape == x.shape, "Shape consistency failed for grad_y"
    
    # Assertion 2: Boundary conditions and math accuracy
    # In a linear slope of 1 per pixel, a standard central difference gradient should be constant (approx 1.0 depending on scaling).
    # Since our Sobel kernel is scaled to give true derivatives (sum of absolute values = 1 for the relevant axis), 
    # it should output roughly 1.0 in the bulk of the slope.
    # At boundary (circular), it wraps from 15 to 0, creating a sharp negative gradient.
    assert grad_x[0, 0, 8, 8] > 0.5, "Math/Boundary failed: expected positive x-gradient on linear slope"
    assert torch.isclose(grad_y[0, 0, 8, 8], torch.tensor(0.0), atol=1e-5), "Math/Boundary failed: expected zero y-gradient on horizontal slope"
    
    # Assertion 3: Gradient stability
    assert x.grad is not None, "Gradient stability failed: missing grad"
    assert not torch.isnan(x.grad).any(), "Gradient stability failed: NaNs in gradient"
    assert not torch.isinf(x.grad).any(), "Gradient stability failed: Infs in gradient"


def test_active_inference_step_invariants():
    # ARRANGE
    size = 16
    model = ActiveInferenceAgent(size=size, dt=0.1, sigma=0.0) # sigma=0 for deterministic grad check
    model.u.requires_grad = True
    model.v.requires_grad = True
    
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
    # The active inference agent has advection and decay. We must ensure it doesn't explode wildly.
    max_change_u = torch.max(torch.abs(u_new - u_init))
    assert max_change_u < 10.0, "Boundary condition failed: update step exploded due to advection or metabolism"
    
    # Assertion 3: Gradient stability
    # The advection requires gradients of gradients (since N is static, but u * grad(N) is used).
    # If N was dynamic, it would be second-order. Since N is static, it's just first order w.r.t u.
    assert u_init.grad is not None, "Gradient stability failed: u missing grad"
    assert v_init.grad is not None, "Gradient stability failed: v missing grad"
    assert not torch.isnan(u_init.grad).any(), "Gradient stability failed: NaNs in u.grad"
    assert not torch.isnan(v_init.grad).any(), "Gradient stability failed: NaNs in v.grad"
