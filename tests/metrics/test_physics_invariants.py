import pytest
import torch
import numpy as np
from src.metrics.metrics import ThermodynamicMetrics

def test_physics_invariant_sine_wave_ksm():
    """
    Generate a perfectly stable 10D biological sine wave (Time=500).
    Assert that the calculate_ksm function returns values consistently > 0.95.
    """
    time_steps = 500
    dim = 10
    
    # Generate stable sine wave
    t = torch.linspace(0, 2 * np.pi, time_steps).unsqueeze(1)
    freqs = torch.linspace(0.5, 2.0, dim)
    stable_signal = torch.sin(t * freqs)
    
    metrics = ThermodynamicMetrics()
    ksm_scores = metrics.calculate_ksm(stable_signal, window_size=50, rank_method="default")
    
    # Check stability after initial window
    for i, ksm in enumerate(ksm_scores[50:]):
        assert ksm > 0.95, f"KSM dropped to {ksm} at {i+50}, expected > 0.95 for stable sine wave"

def test_physics_invariant_gaussian_noise_ksm():
    """
    Generate a 10D tensor of pure Gaussian white noise.
    Assert that the calculate_ksm function returns values < 0.2.
    """
    time_steps = 500
    dim = 10
    
    # Generate pure Gaussian white noise
    torch.manual_seed(42)
    noise_signal = torch.randn(time_steps, dim) * 5.0  # High variance noise
    
    metrics = ThermodynamicMetrics()
    ksm_scores = metrics.calculate_ksm(noise_signal, window_size=50, rank_method="dynamic")
    
    for i, ksm in enumerate(ksm_scores[50:]):
        assert ksm < 0.05, f"KSM at step {i+50} is {ksm}, expected < 0.05 for pure Gaussian white noise"

def test_physics_invariant_ksm_integration():
    """
    Integration test for KSM:
    1. Starts with pure noise (KSM < 0.05)
    2. Transitions to perfectly stable sine wave (KSM > 0.95)
    3. Transitions back to pure noise (KSM < 0.05)
    """
    dim = 10
    window_size = 50
    torch.manual_seed(42)
    
    # Phase 1: Pure Noise
    noise1 = torch.randn(150, dim) * 5.0
    
    # Phase 2: Stable Sine Wave
    t = torch.linspace(0, 2 * np.pi, 500).unsqueeze(1)
    freqs = torch.linspace(0.5, 2.0, dim)
    stable = torch.sin(t * freqs)
    
    # Phase 3: Pure Noise
    noise2 = torch.randn(150, dim) * 5.0
    
    # Concatenate sequence
    signal = torch.cat([noise1, stable, noise2], dim=0)
    
    metrics = ThermodynamicMetrics()
    ksm_scores = metrics.calculate_ksm(signal, window_size=window_size, rank_method="dynamic")
    
    # Check Phase 1 (frames 50 to 150) -> noise
    for i, ksm in enumerate(ksm_scores[50:150]):
        assert ksm < 0.05, f"Phase 1 error: Expected KSM < 0.05 during initial noise, got {ksm}"
        
    # Check Phase 2 (frames 200 to 650) -> stable
    # We start at 200 to give it 50 frames to flush out the Phase 1 noise from the sliding window
    for i, ksm in enumerate(ksm_scores[200:650]):
        assert ksm > 0.95, f"Phase 2 error: Expected KSM > 0.95 during stable signal, got {ksm}"
        
    # Check Phase 3 (frames 700 to 800) -> noise
    # We start at 700 to give it 50 frames to flush out the Phase 2 stable signal
    for i, ksm in enumerate(ksm_scores[700:]):
        assert ksm < 0.05, f"Phase 3 error: Expected KSM < 0.05 during final noise, got {ksm}"

def test_physics_invariant_ksm_middle_case():
    """
    Test that a semi-stable signal (sine wave + moderate noise)
    yields a KSM between 0.4 and 0.7.
    """
    time_steps = 500
    dim = 10
    torch.manual_seed(42)
    
    t = torch.linspace(0, 2 * np.pi, time_steps).unsqueeze(1)
    freqs = torch.linspace(0.5, 2.0, dim)
    stable_signal = torch.sin(t * freqs)
    
    # Add moderate noise to create a ~50% stability case
    noise = torch.randn(time_steps, dim) * 1.5
    mixed_signal = stable_signal + noise
    
    metrics = ThermodynamicMetrics()
    ksm_scores = metrics.calculate_ksm(mixed_signal, window_size=50, rank_method="dynamic")
    
    mean_ksm = np.mean(ksm_scores[50:])
    assert 0.4 < mean_ksm < 0.7, f"Expected mean KSM between 0.4 and 0.7 for mixed signal, got {mean_ksm}"
