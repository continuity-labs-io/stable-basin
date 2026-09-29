import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.benchmarks.lifespan.killifish_benchmark import compute_f1_f2, compute_features_for_fish, seed_everything

def test_compute_f1_f2_invariants():
    """
    ARRANGE: Define inputs and constants.
    """
    seed_everything(42)
    
    # Mock valid sessions
    valid_sessions = []
    
    # Session 1: age 50
    session1_features = np.random.randn(100, 3)
    valid_sessions.append({
        'age': 50,
        'features': session1_features,
        'cols': ['c1_m', 'c2_m', 'c3_m']
    })
    
    # Session 2: age 60
    session2_features = np.random.randn(100, 3)
    valid_sessions.append({
        'age': 60,
        'features': session2_features,
        'cols': ['c1_m', 'c2_m', 'c3_m']
    })
    
    """
    ACT: Execute function under test.
    """
    f1, f2 = compute_f1_f2(valid_sessions)
    
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

def test_compute_features_for_fish(tmp_path):
    """
    ARRANGE: Define inputs and constants.
    """
    # Create a mock h5 file
    h5_file = tmp_path / "mock.h5"
    df = pd.DataFrame({
        'feat1_m': [0.1, 0.2, 0.3, 0.4],
        'feat2_s': [1.0, 1.1, 1.2, 1.3],
        'ignore_col': [0, 0, 0, 0]
    })
    df.to_hdf(h5_file, key='data')
    
    fish_sessions = [
        {'file': str(h5_file), 'age': 40}
    ]
    
    """
    ACT: Execute function under test.
    """
    valid_sessions = compute_features_for_fish(fish_sessions, ['feat1_m', 'feat2_s'], 70)
    
    """
    ASSERT: Verify output length and shape.
    """
    assert valid_sessions is not None
    assert len(valid_sessions) == 1
    assert valid_sessions[0]['age'] == 40
    assert valid_sessions[0]['features'].shape == (4, 2)
    
    # Test age > L (should be ignored)
    valid_sessions_ignored = compute_features_for_fish(fish_sessions, ['feat1_m', 'feat2_s'], 30)
    assert valid_sessions_ignored is None
