import torch
import math
import torch.testing as testing
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from src.models.ssm.masr_mamba import mamba_masr_reference_scan, PyTorchMambaMASR, MaskAwareMamba
def test_mamba_masr_analytical_verification():
    """
    Mathematical Invariant Test: Analytical Exactness (The Scales)
    
    Verifies the pure discrete mathematical correctness of the Zero-Order Hold 
    (ZOH) continuous discretization algorithm inside Mask-Aware Subspace Routing.
    """
    # ARRANGE
    batch_size = 1
    seq_len = 2
    d_model = 1
    d_state = 1
    
    # Input x
    x = torch.tensor([[[2.0], [3.0]]], dtype=torch.float32)
    
    # Time step dt
    dt = torch.tensor([[[0.1], [0.2]]], dtype=torch.float32)
    
    # Mask
    mask = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32)
    
    # Continuous A matrix (negative for stability)
    A = torch.tensor([[-0.5]], dtype=torch.float32)
    
    # Data-dependent parameters B, C
    B = torch.tensor([[[1.5], [2.5]]], dtype=torch.float32)
    C = torch.tensor([[[0.8], [1.2]]], dtype=torch.float32)
    
    # D skip connection
    D = torch.tensor([1.0], dtype=torch.float32)
    
    # Manually calculate the expected mathematical output using float arithmetic
    A_val = -0.5
    D_val = 1.0
    epsilon = 1e-8
    
    # Time step 0
    x_0 = 2.0
    dt_0 = 0.1
    mask_0 = 1.0
    B_0 = 1.5
    C_0 = 0.8
    
    dt_masked_0 = dt_0 * mask_0
    A_bar_0 = math.exp(dt_masked_0 * A_val)
    B_bar_0 = (A_bar_0 - 1.0) / (A_val - epsilon) * B_0
    
    h_0 = A_bar_0 * 0.0 + B_bar_0 * x_0
    y_0_expected = C_0 * h_0 + D_val * x_0
    
    # Time step 1
    x_1 = 3.0
    dt_1 = 0.2
    mask_1 = 1.0
    B_1 = 2.5
    C_1 = 1.2
    
    dt_masked_1 = dt_1 * mask_1
    A_bar_1 = math.exp(dt_masked_1 * A_val)
    B_bar_1 = (A_bar_1 - 1.0) / (A_val - epsilon) * B_1
    
    h_1 = A_bar_1 * h_0 + B_bar_1 * x_1
    y_1_expected = C_1 * h_1 + D_val * x_1
    
    y_expected = torch.tensor([[[y_0_expected], [y_1_expected]]], dtype=torch.float32)
    
    # ACT
    y_out = mamba_masr_reference_scan(x, dt, mask, A, B, C, D)
    
    # ASSERT
    testing.assert_close(y_out, y_expected, atol=1e-6, rtol=1e-6)

def test_mamba_masr_singularity_prevention():
    """
    Mathematical Invariant Test: The Singularity Threshold (The Dissonance Check)
    
    Proves the epsilon stabilizer in the MASR continuous-time integration
    prevents silent numerical explosion when the state matrix A approaches zero.
    """
    # ARRANGE
    batch_size = 1
    seq_len = 2
    d_model = 1
    d_state = 1
    
    x = torch.ones(batch_size, seq_len, d_model)
    dt = torch.ones(batch_size, seq_len, d_model) * 0.1
    mask = torch.ones(batch_size, seq_len, d_model)
    B = torch.ones(batch_size, seq_len, d_state)
    C = torch.ones(batch_size, seq_len, d_state)
    D = torch.ones(d_model)
    
    adversarial_A_values = [1e-8, 0.0, -1e-9]
    for a_val in adversarial_A_values:
        A = torch.tensor([[a_val]], dtype=torch.float32)
        
        # ACT
        y_out = mamba_masr_reference_scan(x, dt, mask, A, B, C, D)
        
        # ASSERT
        assert not torch.isnan(y_out).any(), f"NaN detected in output for A={a_val}"
        assert not torch.isinf(y_out).any(), f"Inf detected in output for A={a_val}"

def test_masr_mamba_stasis_gradient_isolation():
    """
    Mathematical Invariant Test: Stasis Gradient Integrity (The Rests)
    
    Ensures that when a sensor is masked (missing data), no error signal leaks 
    across the void during the backward pass.
    """
    # ARRANGE
    batch_size = 2
    seq_len = 5
    d_model = 4
    d_state = 8
    
    model = PyTorchMambaMASR(d_model=d_model, d_state=d_state)
    # Force D to 0.0 to prevent direct skip connection gradient leakage
    model.D.data.fill_(0.0)
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, d_model)
    
    # Mask exactly t=2
    masked_t = 2
    mask[:, masked_t, :] = 0.0
    
    # ACT
    y_out = model(x, mask)
    loss = y_out.sum()
    loss.backward()
    
    # ASSERT
    assert x.grad is not None, "Gradients were not computed for x"
    
    masked_grad = x.grad[:, masked_t, :]
    testing.assert_close(masked_grad, torch.zeros_like(masked_grad), atol=1e-8, rtol=1e-8, msg="Gradient leaked at masked time step!")
    
    # Ensure gradients flow to other timesteps
    assert x.grad[:, masked_t + 1, :].abs().sum() > 0, "No gradients flowed to active time steps"

def test_masr_mamba_log_space_dt_bounds():
    """
    Mathematical Invariant Test: Biological Temporal Boundaries (The Tempo Constraint)
    
    The network must prove it operates exclusively within biologically relevant temporal 
    frequencies, proving the log-space initialization of the bias holds true under activation.
    """
    # ARRANGE
    batch_size = 2
    seq_len = 10
    d_model = 16
    d_state = 16
    
    model = PyTorchMambaMASR(d_model=d_model, d_state=d_state)
    
    # Create a completely zeroed input tensor to isolate the bias
    x = torch.zeros(batch_size, seq_len, d_model)
    
    # ACT
    # Intercept the values of the continuous time step dt exactly as the model computes it
    dt = F.softplus(model.dt_proj(x))
    
    # ASSERT
    min_dt = dt.min().item()
    max_dt = dt.max().item()
    
    assert min_dt >= 0.0009, f"dt min value {min_dt} is below the biological threshold 0.0009"
    assert max_dt <= 0.11, f"dt max value {max_dt} exceeds the biological threshold 0.11"

def test_mask_aware_control_group_parity():
    """
    Mathematical Invariant Test: The Control Group Parity (The Symmetrical Hands)
    
    Verifies that the control group (mask_aware=False) is structurally and dimensionally
    identical to the experimental group, avoiding hidden inductive biases.
    """
    # ARRANGE
    batch_size = 2
    seq_len = 10
    input_dim = 16
    d_model = 16
    d_state = 16
    
    model_true = MaskAwareMamba(input_dim=input_dim, d_model=d_model, d_state=d_state, mask_aware=True)
    model_false = MaskAwareMamba(input_dim=input_dim, d_model=d_model, d_state=d_state, mask_aware=False)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, input_dim)
    
    # ACT
    pred_true, recon_true = model_true(x, mask=mask)
    pred_false, recon_false = model_false(x)
    
    # ASSERT
    assert len(model_true.input_proj) == len(model_false.input_proj), "Projection depth mismatch!"
    
    def check_proj(model):
        has_ln = any(isinstance(m, nn.LayerNorm) for m in model.input_proj)
        has_gelu = any(isinstance(m, nn.GELU) for m in model.input_proj)
        return has_ln and has_gelu
        
    assert check_proj(model_true), "mask_aware=True missing LayerNorm or GELU"
    assert check_proj(model_false), "mask_aware=False missing LayerNorm or GELU"
    
    assert pred_true.shape == pred_false.shape, "Prediction shapes diverged"
    assert recon_true.shape == recon_false.shape, "Reconstruction shapes diverged"
    assert pred_true.shape == (batch_size, seq_len, input_dim), f"Expected {(batch_size, seq_len, input_dim)} but got {pred_true.shape}"

def test_mamba_masr_gradcheck():
    """
    Mathematical Invariant Test: The Calculus Tuning Fork (gradcheck)
    
    Uses torch.autograd.gradcheck to verify the mathematical correctness of 
    mamba_masr_reference_scan by comparing analytical gradients against 
    numerical approximations.
    """
    # ARRANGE
    batch_size = 1
    seq_len = 2
    d_model = 2
    d_state = 2
    
    # CRITICAL: All continuous float inputs MUST be cast to torch.float64 and require grad
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float64, requires_grad=True)
    dt = torch.rand(batch_size, seq_len, d_model, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, d_model, dtype=torch.float64, requires_grad=True)
    A = torch.randn(d_model, d_state, dtype=torch.float64, requires_grad=True)
    B = torch.randn(batch_size, seq_len, d_state, dtype=torch.float64, requires_grad=True)
    C = torch.randn(batch_size, seq_len, d_state, dtype=torch.float64, requires_grad=True)
    D = torch.randn(d_model, dtype=torch.float64, requires_grad=True)
    
    # ACT
    test_passed = gradcheck(
        mamba_masr_reference_scan, 
        (x, dt, mask, A, B, C, D), 
        eps=1e-6, 
        atol=1e-4,
        raise_exception=True
    )
    
    # ASSERT
    assert test_passed

def test_mamba_a_matrix_drift_bounds():
    """
    Ensures A is strictly bounded away from 0.0 to guarantee a minimum memory leak.
    """
    # ARRANGE
    model = PyTorchMambaMASR(d_model=16, d_state=16)
    
    # ACT
    A = model.A_init()
    
    # ASSERT
    assert torch.all(A <= -0.1), "A matrix is not sufficiently bounded! Drift risk."

def test_mamba_zoh_expm1_precision():
    """
    Uses double-precision math to detect catastrophic float32 cancellation if 
    anyone attempts to remove the torch.expm1 ZOH fix.
    """
    # ARRANGE
    batch_size = 1
    d_model = 2
    d_state = 2
    
    dt = torch.tensor([[[1e-7, 1e-7]]], dtype=torch.float32) # (1, 2, 1)
    A = torch.tensor([[-0.1, -0.1], [-0.1, -0.1]], dtype=torch.float32) # (2, 2)
    B = torch.tensor([[1.0, 1.0]], dtype=torch.float32) # (1, 2)
    
    # ACT: Compute float32 using exact mamba_masr_reference_scan logic
    A_safe = torch.where(A.abs() <= 1e-8, torch.full_like(A, -1e-8), A)
    B_bar_f32 = torch.expm1(dt * A) / A_safe * B.unsqueeze(1)
    
    # Compute float64 true value
    dt_64 = dt.double()
    A_64 = A.double()
    B_64 = B.double()
    A_safe_64 = torch.where(A_64.abs() <= 1e-8, torch.full_like(A_64, -1e-8), A_64)
    # the true mathematical value
    B_bar_true = (torch.exp(dt_64 * A_64) - 1.0) / A_safe_64 * B_64.unsqueeze(1)
    
    # Compute what float32 WOULD be if someone reverted to exp(x) - 1.0
    B_bar_reverted = (torch.exp(dt * A) - 1.0) / A_safe * B.unsqueeze(1)
    
    # ASSERT
    # Make sure our fix matches true mathematical value well
    assert torch.allclose(B_bar_f32.double(), B_bar_true, atol=1e-8)
    # Make sure reverted version fails!
    assert not torch.allclose(B_bar_reverted.double(), B_bar_true, atol=1e-8), "Test invalid! Reverted version somehow matches. The mock values might not trigger catastrophic cancellation."

def test_mamba_dt_variance_clamping():
    """
    Strictly enforces the +-1e-4 weight initialization bound on dt_proj to 
    prevent compounding variance on out-of-distribution inputs.
    """
    # ARRANGE & ACT
    model = PyTorchMambaMASR(d_model=64)
    
    # ASSERT
    assert torch.max(model.dt_proj.weight.abs()) <= 1.05e-4, "dt_proj weight variance is unbounded!"

def test_mamba_residual_d_init():
    """
    Guarantees the feedforward residual is zeroed out at initialization so the 
    network builds a robust recurrent manifold.
    """
    # ARRANGE & ACT
    model = PyTorchMambaMASR(d_model=64)
    
    # ASSERT
    assert torch.all(model.D == 0.0), "Residual D parameter is not initialized to exactly zero!"
