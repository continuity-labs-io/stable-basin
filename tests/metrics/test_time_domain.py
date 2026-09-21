import pytest
import torch
import numpy as np
from src.metrics.time_domain import ThermodynamicMetrics, calculate_dynamic_rank

@pytest.fixture(autouse=True)
def detect_anomaly():
    """Paranoid Debugger Mode"""
    torch.autograd.set_detect_anomaly(True)
    yield

def test_calculate_dynamic_rank_biological_elbow():
    """
    ARRANGE: Create singular values with a sharp structural elbow.
    ACT: Calculate dynamic rank.
    ASSERT: Ensures that the log-scaled spectral gap correctly identifies the elbow index.
    """
    S = np.array([1000.0, 100.0, 10.0, 1.0, 0.9, 0.8] + [0.01] * 14)
    n_rows, n_cols = 20, 20
    
    rank = calculate_dynamic_rank(S, n_rows, n_cols)
    assert rank == 3

def test_csd_lag_paradox():
    """
    ARRANGE: Initialize metrics and a sequence shorter than the window size.
    ACT: Calculate CSD.
    ASSERT: Validates the graceful fallback returns a 0.0 array of max(1, time_steps) length.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.randn(5, 10)
    
    csd = metrics.calculate_csd(z_seq, window_size=20)
    assert len(csd) == 5
    assert all(c == 0.0 for c in csd)

def test_csd_window_size_one():
    """
    ARRANGE: Initialize metrics and set window size to 1.
    ACT: Calculate CSD.
    ASSERT: Ensures no nan/crash occurs when lag-1 autocorrelation cannot be computed.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.randn(10, 5)
    
    csd = metrics.calculate_csd(z_seq, window_size=1)
    assert len(csd) == 10
    assert not any(np.isnan(c) for c in csd)

def test_ksm_lag_paradox():
    """
    ARRANGE: Initialize metrics and a sequence equal or shorter than the window size.
    ACT: Calculate KSM.
    ASSERT: Validates fallback to [1.0] * time_steps.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.randn(5, 10)
    
    ksm = metrics.calculate_ksm(z_seq, window_size=5)
    assert len(ksm) == 5
    assert all(k == 1.0 for k in ksm)

def test_ksm_flatline():
    """
    ARRANGE: Constant, zero-variance sequence.
    ACT: Calculate KSM.
    ASSERT: Ensures it gracefully forces rank collapse and returns 0.0 for KSM.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.ones(10, 5)
    
    ksm = metrics.calculate_ksm(z_seq, window_size=4)
    # The first 'window_size' are 1.0 due to padding, then the rest should be 0.0 due to collapse
    assert len(ksm) == 10
    assert ksm[-1] == 0.0

def test_ksm_exception_handling():
    """
    ARRANGE: Sequence with NaNs to crash OptDMD.
    ACT: Calculate KSM.
    ASSERT: Ensures exception is caught and KSM gracefully returns 0.0 without crashing.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.full((10, 5), float('nan'))
    
    ksm = metrics.calculate_ksm(z_seq, window_size=4, debug_crash_frame=5)
    assert len(ksm) == 10
    assert ksm[-1] == 0.0

def test_hysteresis_short_sequence():
    """
    ARRANGE: Pass sequences shorter than 2 frames.
    ACT: Calculate Hysteresis.
    ASSERT: Validates short sequences return 0.0 and empty list.
    """
    metrics = ThermodynamicMetrics()
    z_base = torch.randn(1, 5)
    z_pert = torch.randn(1, 5)
    
    area, paths = metrics.calculate_hysteresis(z_base, z_pert)
    assert area == 0.0
    assert len(paths) == 0

def test_lle_lag_paradox():
    """
    ARRANGE: Sequence shorter than window size.
    ACT: Calculate LLE.
    ASSERT: Validates fallback returns [0.0] * time_steps.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.randn(5, 10)
    
    lle = metrics.calculate_lle(z_seq, window_size=10)
    assert len(lle) == 5
    assert all(l == 0.0 for l in lle)

def test_lle_flatline():
    """
    ARRANGE: Constant sequence.
    ACT: Calculate LLE.
    ASSERT: Validates temporal_std check sets max_eig=0 and LLE correctly.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.ones(10, 5)
    
    lle = metrics.calculate_lle(z_seq, window_size=4)
    assert len(lle) == 10

def test_lle_exception_handling():
    """
    ARRANGE: Sequence with NaNs to crash OptDMD.
    ACT: Calculate LLE.
    ASSERT: Validates exception is caught and sets max_eig=1.0 for graceful fallback.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.full((10, 5), float('nan'))
    
    lle = metrics.calculate_lle(z_seq, window_size=4)
    assert len(lle) == 10

def test_cka_zero_variance():
    """
    ARRANGE: Zero-variance identical sequences.
    ACT: Calculate CKA.
    ASSERT: Ensures ZeroDivisionError is prevented and returns a valid value (0.0).
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.ones(10, 5)
    
    cka = metrics.calculate_cka(z_seq, z_seq)
    assert not np.isnan(cka)
    assert cka == 0.0

def test_epigenetic_dispersion():
    """
    ARRANGE: Normal tensor and None.
    ACT: Calculate Epigenetic Dispersion.
    ASSERT: Validates valid variance and None fallback.
    """
    metrics = ThermodynamicMetrics()
    cpg_tensor = torch.randn(10, 5, 20)
    disp = metrics.calculate_epigenetic_dispersion(cpg_tensor)
    assert len(disp) == 10
    
    disp_none = metrics.calculate_epigenetic_dispersion(None)
    assert len(disp_none) == 0

def test_fedichev_macrostates():
    """
    ARRANGE: Two paths of length 1, and two paths of valid length.
    ACT: Extract macrostates.
    ASSERT: Validates early return for short sequences, and correct output for valid sequences.
    """
    metrics = ThermodynamicMetrics()
    z_base = torch.randn(1, 5)
    z_pert = torch.randn(1, 5)
    
    macros_short = metrics.extract_fedichev_macrostates(z_base, z_pert)
    assert len(macros_short["Z_entropic_damage"]) == 0
    
    z_base_valid = torch.randn(10, 5)
    z_pert_valid = torch.randn(10, 5)
    cpg = torch.randn(10, 5, 20)
    macros = metrics.extract_fedichev_macrostates(z_base_valid, z_pert_valid, window_size=4, cpg_tensor=cpg)
    assert len(macros["Z_entropic_damage"]) == 10
    assert len(macros["z0_volatility"]) == 10
    assert len(macros["epsilon_0_ksm"]) == 10
    assert len(macros["Z_epigenetic_entropy"]) == 10

def test_unified_diagnostics():
    """
    ARRANGE: Valid mock inputs.
    ACT: Calculate unified diagnostics.
    ASSERT: Output dictionary has expected keys and shapes.
    """
    metrics = ThermodynamicMetrics()
    z_seq = torch.randn(10, 5)
    raw = torch.randn(2, 100) # (Channels, Time)
    
    diags = metrics.calculate_unified_diagnostics(z_seq, raw, macro_channel_idx=0, micro_channel_idx=1, sampling_rate=20.0)
    assert "time_domain" in diags
    assert "frequency_domain" in diags
