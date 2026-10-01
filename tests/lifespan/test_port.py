import os
import json
import pytest
import numpy as np
from pathlib import Path
from src.benchmarks.lifespan.adapters import ADAPTERS
from src.benchmarks.lifespan.landmark import cut_at_landmark
from src.features.static import static
from src.features.dynamics import dynamics
from src.benchmarks.lifespan.evaluate import run_evaluation_pipeline

@pytest.mark.skipif(not Path('data/killifish/data/b0_20250415/df_reformat_b0_10_20250416.csv').exists(), reason="Killifish data missing")
def test_port():
    cfg = {
        'path': 'data/killifish/data/b0_20250415/df_reformat_b0_10_20250416.csv',
        'filters': {'genotype': 'wt', 'sex': 'male', 'feeding': 'al'},
        'negative_is_missing': False
    }
    
    animals, series, variables = ADAPTERS['killifish_bedbrook'](cfg)
    L = 70
    min_valid_days = 14
    
    pre_series, outcomes = cut_at_landmark(animals, series, variables, L, min_valid_days)
    
    X_f1_list = []
    X_f2_list = []
    
    # Needs to be sorted or same order as original to match exactly?
    # Actually evaluate pipeline does cross-validation, so the exact means will depend on the order of items.
    # The original script processes fish in order of appearance in df.
    # `animals` is already ordered by first appearance in source file.
    for aid in outcomes['animal_id']:
        rows = pre_series[pre_series['animal_id'] == aid]
        f_s = static(rows, variables)
        X_f1_list.append(f_s.values)
        
        f_d = dynamics(rows, variables)
        X_f2_list.append(f_d.values)
        
    X_f1 = np.array(X_f1_list)
    X_f2 = np.array(X_f2_list)
    X_both = np.hstack([X_f1, X_f2])
    
    Y_T = outcomes['T'].values
    Y_E = outcomes['E'].values
    
    c_f1, _, _ = run_evaluation_pipeline(X_f1, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10)
    c_f2, _, _ = run_evaluation_pipeline(X_f2, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10)
    c_both, _, _ = run_evaluation_pipeline(X_both, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10)
    
    assert abs(np.mean(c_f1) - 0.4645833) < 0.01
    assert abs(np.mean(c_f2) - 0.4364583) < 0.01
    assert abs(np.mean(c_both) - 0.4579427) < 0.01
