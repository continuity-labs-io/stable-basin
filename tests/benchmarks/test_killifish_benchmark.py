import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.benchmarks.lifespan.killifish_benchmark import compute_f1_f2_for_session, get_fish_features, seed_everything, run_evaluation_pipeline

def test_compute_f1_f2_invariants():
    """
    ARRANGE: Define inputs and constants.
    """
    seed_everything(42)
    
    # Mock valid sessions
    valid_sessions = []
    
    # Session 1: age 50
    session1_features = np.random.randn(100, 3)
    feats1 = compute_f1_f2_for_session(session1_features)
    valid_sessions.append({
        'age': 50,
        'cached_features': feats1
    })
    
    # Session 2: age 60
    session2_features = np.random.randn(100, 3)
    feats2 = compute_f1_f2_for_session(session2_features)
    valid_sessions.append({
        'age': 60,
        'cached_features': feats2
    })
    
    """
    ACT: Execute function under test.
    """
    f1, f2 = get_fish_features(valid_sessions)
    
    """
    ASSERT: Verify boundaries and shapes.
    """
    # f1 should be [f1_mean (3 features), f1_sd (3 features)] -> 6 length
    assert f1.shape == (6,)
    assert not np.isnan(f1).any()
    
    # f2 should be [f2_mean (9 features), f2_slope (9 features)] -> 18 length
    # f2_session_features has lag1_acf, var_X, iact for each of 3 features
    assert f2.shape == (18,)
    assert not np.isnan(f2).any()

def test_fft_iact_matches_direct():
    """
    ARRANGE: Define inputs and constants.
    """
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 2)
    
    # Direct loop
    mean_X = np.mean(X, axis=0)
    t = np.arange(n)
    t = t - np.mean(t)
    var_t = np.sum(t ** 2)
    cov = np.sum(t[:, None] * (X - mean_X), axis=0)
    slope = cov / var_t
    X_detrended = X - mean_X - slope * t[:, None]
    
    iact_direct = np.zeros(2)
    for f_idx in range(2):
        x_f = X_detrended[:, f_idx]
        var_xf = np.sum(x_f ** 2)
        acf_sum = 0
        for k in range(1, min(500, n)):
            cov_k = np.sum(x_f[:-k] * x_f[k:])
            acf_k = cov_k / var_xf
            if acf_k <= 0:
                break
            acf_sum += acf_k
        iact_direct[f_idx] = 1 + 2 * acf_sum
        
    """
    ACT: Execute function under test.
    """
    feats = compute_f1_f2_for_session(X)
    
    """
    ASSERT: Verify output matches.
    """
    # feats order: mean_X (2), lag1_acf (2), var_X (2), iact (2)
    iact_fft = feats[6:8]
    assert np.allclose(iact_direct, iact_fft, atol=1e-6)

def test_synthetic_coxph_positive():
    """
    ARRANGE: Define inputs and constants.
    """
    np.random.seed(42)
    n_fish = 100
    X_both = np.random.randn(n_fish, 4)
    Y_T = 100 - 10 * X_both[:, 0] + np.random.randn(n_fish) * 2
    Y_T = np.maximum(1, Y_T)
    Y_E = np.ones(n_fish)
    
    """
    ACT: Execute function under test.
    """
    c_indices, _ = run_evaluation_pipeline(X_both, Y_T, Y_E, repeats=5, n_splits=5, seed=42)
    
    """
    ASSERT: Verify G1 passes.
    """
    gate1_pass = np.percentile(c_indices, 2.5) > 0.5
    assert gate1_pass, f"Gate 1 failed, 2.5th percentile was {np.percentile(c_indices, 2.5)}"

def test_synthetic_coxph_null():
    """
    ARRANGE: Define inputs and constants.
    """
    np.random.seed(42)
    n_fish = 100
    X_both = np.random.randn(n_fish, 4)
    Y_T = 100 - 10 * X_both[:, 0] + np.random.randn(n_fish) * 2
    Y_T = np.maximum(1, Y_T)
    Y_E = np.ones(n_fish)
    
    """
    ACT: Execute function under test.
    """
    c_indices, _ = run_evaluation_pipeline(X_both, Y_T, Y_E, repeats=5, n_splits=5, seed=42, shuffle_y=True)
    
    """
    ASSERT: Verify C-index within [0.45, 0.55].
    """
    mean_c = np.mean(c_indices)
    assert 0.45 <= mean_c <= 0.55, f"Mean C-index {mean_c} out of bounds"
