import numpy as np
from src.benchmarks.lifespan.adapters import ADAPTERS
from src.benchmarks.lifespan.landmark import cut_at_landmark
from src.features.static import static
from src.features.dynamics import dynamics
from src.benchmarks.lifespan.evaluate import run_evaluation_pipeline

def test_positive_control():
    cfg = {}
    animals, series, variables = ADAPTERS['synthetic_positive'](cfg)
    L = 60
    min_valid_days = 14
    
    pre_series, outcomes = cut_at_landmark(animals, series, variables, L, min_valid_days)
    X_dyn_list = []
    
    for aid in outcomes['animal_id']:
        rows = pre_series[pre_series['animal_id'] == aid]
        f_d = dynamics(rows, variables)
        X_dyn_list.append(f_d.values)
        
    X_f2 = np.array(X_dyn_list)
    Y_T = outcomes['T'].values
    Y_E = outcomes['E'].values
    
    c_f2, _, _ = run_evaluation_pipeline(X_f2, Y_T, Y_E, repeats=5, n_splits=5, seed=42, n_components=10)
    assert np.mean(c_f2) >= 0.65

def test_negative_control():
    cfg = {}
    animals, series, variables = ADAPTERS['synthetic_after_landmark_only'](cfg)
    L = 60
    min_valid_days = 14
    
    pre_series, outcomes = cut_at_landmark(animals, series, variables, L, min_valid_days)
    X_both_list = []
    
    for aid in outcomes['animal_id']:
        rows = pre_series[pre_series['animal_id'] == aid]
        f_s = static(rows, variables)
        f_d = dynamics(rows, variables)
        X_both_list.append(np.concatenate([f_s.values, f_d.values]))
        
    X_both = np.array(X_both_list)
    Y_T = outcomes['T'].values
    Y_E = outcomes['E'].values
    
    c_both, _, _ = run_evaluation_pipeline(X_both, Y_T, Y_E, repeats=5, n_splits=5, seed=42, n_components=10)
    assert 0.35 <= np.mean(c_both) <= 0.65
