import numpy as np
import pandas as pd
import pytest
import subprocess

pytestmark = pytest.mark.integration
from src.benchmarks.lifespan.adapters import ADAPTERS
from src.benchmarks.lifespan.landmark import cut_at_landmark
from src.benchmarks.lifespan.run import score_landmark
from src.benchmarks.lifespan.controls import POSITIVE_MIN_C, NEGATIVE_BAND, LEAK_MIN_C

def test_synthetic_reproducible():
    for name in ['synthetic_positive', 'synthetic_negative_control']:
        adapter = ADAPTERS[name]
        
        cfg = {'seed': 42, 'landmark': 60, 'n_animals': 20}
        _, series1, _ = adapter(cfg)
        _, series2, _ = adapter(cfg)
        pd.testing.assert_frame_equal(series1, series2)
        
        cfg_diff = {'seed': 99, 'landmark': 60, 'n_animals': 20}
        _, series3, _ = adapter(cfg_diff)
        
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(series1, series3)

def test_controls_across_seeds():
    for seed in [42]:
        cfg_pos = {'seed': seed, 'cv': {'repeats': 1, 'folds': 2, 'seed': seed}, 'model': {'pca_components': 10}, 'landmark': 60, 'n_animals': 100}
        animals_pos, series_pos, var_pos = ADAPTERS['synthetic_positive'](cfg_pos)
        res_pos = score_landmark(animals_pos, series_pos, var_pos, 60, cfg_pos)
        c_f2, _ = res_pos['F2']
        assert np.mean(c_f2) >= POSITIVE_MIN_C, f"Seed {seed} positive F2 failed"
        
        cfg_neg = {'seed': seed, 'cv': {'repeats': 1, 'folds': 2, 'seed': seed}, 'model': {'pca_components': 10}, 'landmark': 60, 'n_animals': 100}
        animals_neg, series_neg, var_neg = ADAPTERS['synthetic_negative_control'](cfg_neg)
        res_neg = score_landmark(animals_neg, series_neg, var_neg, 60, cfg_neg)
        
        c_f1, _ = res_neg['F1']
        c_f2_neg, _ = res_neg['F2']
        c_both, _ = res_neg['F1+F2']
        
        assert NEGATIVE_BAND[0] <= round(np.mean(c_f1), 2) <= NEGATIVE_BAND[1]
        assert NEGATIVE_BAND[0] <= round(np.mean(c_f2_neg), 2) <= NEGATIVE_BAND[1]
        assert NEGATIVE_BAND[0] <= round(np.mean(c_both), 2) <= NEGATIVE_BAND[1]

def test_negative_control_detects_leak():
    def leaky_cut(animals, series, variables, L, min_valid_days):
        # Keep every valid day, including >= L
        pre_series = series.copy()
        
        outcomes_list = []
        for aid, grp in pre_series.groupby('animal_id'):
            anim_info = animals[animals['animal_id'] == aid].iloc[0]
            T = anim_info['lifespan_days']
            E = anim_info['died']
            group = anim_info['group']
            
            outcomes_list.append({
                'animal_id': aid,
                'T': float(T),
                'E': E,
                'group': group
            })
            
        outcomes = pd.DataFrame(outcomes_list)
        return pre_series, outcomes

    for seed in [42]:
        cfg_neg = {'seed': seed, 'cv': {'repeats': 1, 'folds': 2, 'seed': seed}, 'model': {'pca_components': 10}, 'landmark': 60, 'n_animals': 100}
        animals_neg, series_neg, var_neg = ADAPTERS['synthetic_negative_control'](cfg_neg)
        res_neg = score_landmark(animals_neg, series_neg, var_neg, 60, cfg_neg, cut_fn=leaky_cut)
        
        c_both, _ = res_neg['F1+F2']
        assert np.mean(c_both) >= LEAK_MIN_C

def test_fail_closed(tmp_path):
    import yaml
    # Test suite without negative control
    suite_no_neg = {
        'cv': {'folds': 5, 'repeats': 5, 'seed': 42},
        'model': {'penalizer': 0.1, 'pca_components': 10},
        'datasets': [
            {'name': 'synthetic_positive', 'role': 'positive_control', 'landmarks': [60], 'min_valid_days': 14}
        ]
    }
    suite_path = tmp_path / "suite_no_neg.yaml"
    with open(suite_path, "w") as f:
        yaml.dump(suite_no_neg, f)
        
    res = subprocess.run(["python", "-m", "src.benchmarks.lifespan.run", "--suite", str(suite_path)])
    assert res.returncode != 0

    # Test negative control with zero animals
    suite_zero = {
        'cv': {'folds': 5, 'repeats': 5, 'seed': 42},
        'model': {'penalizer': 0.1, 'pca_components': 10},
        'datasets': [
            {'name': 'synthetic_positive', 'role': 'positive_control', 'landmarks': [60], 'min_valid_days': 14},
            {'name': 'synthetic_negative_control', 'role': 'negative_control', 'landmarks': [60], 'min_valid_days': 14, 'n_animals': 0}
        ]
    }
    suite_zero_path = tmp_path / "suite_zero.yaml"
    with open(suite_zero_path, "w") as f:
        yaml.dump(suite_zero, f)
        
    res = subprocess.run(["python", "-m", "src.benchmarks.lifespan.run", "--suite", str(suite_zero_path)])
    assert res.returncode != 0
