import numpy as np
from src.benchmarks.lifespan.adapters import ADAPTERS
from src.benchmarks.lifespan.landmark import cut_at_landmark
from src.features.static import static
from src.features.dynamics import dynamics

def test_landmark_invariance():
    cfg = {}
    animals, series, variables = ADAPTERS['synthetic_positive'](cfg)
    
    L = 60
    min_valid_days = 14
    
    def get_features(animals, series):
        pre_series, outcomes = cut_at_landmark(animals, series, variables, L, min_valid_days)
        f_list = []
        for aid in sorted(outcomes['animal_id']):
            rows = pre_series[pre_series['animal_id'] == aid]
            f_s = static(rows, variables)
            f_d = dynamics(rows, variables)
            f_list.append(np.concatenate([f_s.values, f_d.values]))
        return np.array(f_list)

    # 1. Original
    f1 = get_features(animals, series)
    
    # 2. Overwrite rows with age_days >= L
    series2 = series.copy()
    mask = series2['age_days'] >= L
    for v in variables:
        series2.loc[mask, v] = np.random.randn(mask.sum())
    f2 = get_features(animals, series2)
    
    # 3. Drop rows with age_days >= L
    series3 = series[series['age_days'] < L].copy()
    f3 = get_features(animals, series3)
    
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(f1, f3)
