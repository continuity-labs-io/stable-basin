import torch
import math
import pytest
from src.metrics import SpectralMetrics
from src.metrics.spectral import _hilbert_transform

@pytest.fixture
def hardware_monitor():
    """Fixture to provide the optimal hardware device and a memory tracking closure."""
    device = torch.device('cpu')
    if torch.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        
    def get_mem():
        if device.type == 'mps':
            return torch.mps.current_allocated_memory()
        elif device.type == 'cuda':
            return torch.cuda.memory_allocated()
        return 0
        
    return device, get_mem

def test_identity_ground_truth():
    """
    The Identity Test (Ground Truth Calibration):
    - Generate a pure 2Hz sine wave and a pure 10Hz sine wave.
    - Assert that PSD accurately identifies the 2Hz and 10Hz bins.
    - Assert that PLV between two identical 2Hz waves is exactly 1.0.
    - Assert that PLV between the 2Hz and 10Hz wave is near 0.0.
    """
    # ARRANGE
    metrics = SpectralMetrics()
    fs = 100.0
    t = torch.linspace(0, 5.0, int(5.0 * fs))
    
    wave_2hz = torch.sin(2 * math.pi * 2.0 * t)
    wave_10hz = torch.sin(2 * math.pi * 10.0 * t)
    
    # ACT
    freqs_2, power_2 = metrics.calculate_psd(wave_2hz, sampling_rate=fs)
    freqs_10, power_10 = metrics.calculate_psd(wave_10hz, sampling_rate=fs)
    
    plv_identical = metrics.calculate_plv(wave_2hz, wave_2hz)
    plv_different = metrics.calculate_plv(wave_2hz, wave_10hz)
    
    # ASSERT
    idx_max_2 = torch.argmax(power_2).item()
    idx_max_10 = torch.argmax(power_10).item()
    
    assert math.isclose(freqs_2[idx_max_2].item(), 2.0, abs_tol=0.5), "PSD failed to identify 2Hz bin."
    assert math.isclose(freqs_10[idx_max_10].item(), 10.0, abs_tol=0.5), "PSD failed to identify 10Hz bin."
    
    assert math.isclose(plv_identical.item() if isinstance(plv_identical, torch.Tensor) else plv_identical, 1.0, abs_tol=1e-4), f"PLV between identical waves should be 1.0, got {plv_identical}"
    
    val_diff = plv_different.item() if isinstance(plv_different, torch.Tensor) else plv_different
    assert val_diff < 0.1, f"PLV between 2Hz and 10Hz should be near 0.0, got {val_diff}"

def test_enslavement_cfc_validation():
    """
    The Enslavement Test (CFC Validation):
    - Generate a synthetic signal where a 50Hz high-frequency burst only occurs during the peak of a 2Hz low-frequency wave.
    - Assert that calculate_cfc_pac returns a strongly positive value (> 0.8).
    - Generate a control signal where the 50Hz bursts are uniformly distributed. Assert CFC is near 0.0.
    """
    # ARRANGE
    metrics = SpectralMetrics()
    fs = 500.0
    t = torch.linspace(0, 10.0, int(10.0 * fs))
    
    slow_wave = torch.sin(2 * math.pi * 2.0 * t)
    fast_carrier = torch.sin(2 * math.pi * 50.0 * t)
    
    envelope_coupled = torch.where(slow_wave > 0.8, 10.0, 0.0)
    fast_coupled = fast_carrier * envelope_coupled
    
    envelope_control = torch.ones_like(t) * envelope_coupled.mean()
    fast_control = fast_carrier * envelope_control
    
    # ACT
    cfc_coupled = metrics.calculate_cfc_pac(slow_wave, fast_coupled)
    cfc_control = metrics.calculate_cfc_pac(slow_wave, fast_control)
    
    # ASSERT
    assert cfc_coupled.item() > 0.8, f"CFC for strongly coupled signal should be > 0.8, got {cfc_coupled.item()}"
    assert cfc_control.item() < 0.1, f"CFC for uncoupled control should be near 0.0, got {cfc_control.item()}"

@pytest.mark.integration
def test_vram_leak_hardware_scaling(hardware_monitor):
    """
    The VRAM Leak Test (Hardware Scaling):
    - Feed a massive tensor (e.g., Batch=8, Channels=1024, Time=5000) into the PLV and CFC functions.
    - Monitor memory footprint.
    - Assert that the memory footprint remains stable and O(1) without accumulating graph history.
    """
    # ARRANGE
    metrics = SpectralMetrics()
    device, get_mem = hardware_monitor
    
    # Time=5000, Batch=8, Channels=1024 (Assuming Time is dimension 0)
    tensor_a = torch.randn(5000, 8, 1024, device=device)
    tensor_b = torch.randn(5000, 8, 1024, device=device)
    
    initial_mem = get_mem()
    
    # ACT - Warmup
    _ = metrics.calculate_plv(tensor_a, tensor_b)
    _ = metrics.calculate_cfc_pac(tensor_a, tensor_b)
    
    post_warmup_mem = get_mem()
    
    for _ in range(5):
        _ = metrics.calculate_plv(tensor_a, tensor_b)
        _ = metrics.calculate_cfc_pac(tensor_a, tensor_b)
        
    final_mem = get_mem()
    
    # ASSERT
    # Expect memory to not grow significantly after warmup
    assert final_mem <= post_warmup_mem + (1024 * 1024), "VRAM leak detected! Memory grew significantly after warmup."

def test_native_hilbert_transform():
    """
    Test that the native PyTorch _hilbert_transform correctly computes the analytic signal.
    For a real cosine wave cos(wt), the analytic signal is cos(wt) + j*sin(wt).
    """
    # ARRANGE
    t = torch.arange(100, dtype=torch.float64) / 100.0
    freq = 5.0
    x = torch.cos(2 * math.pi * freq * t)
    
    # ACT
    actual_analytic = _hilbert_transform(x, dim=0)
    
    # ASSERT
    # The real part should be exactly the original signal
    assert torch.allclose(actual_analytic.real, x, atol=1e-5)
    
    # The imaginary part should be sin(wt)
    expected_imag = torch.sin(2 * math.pi * freq * t)
    assert torch.allclose(actual_analytic.imag, expected_imag, atol=1e-5)
