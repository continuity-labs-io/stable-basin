import argparse
import yaml
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from src.benchmarks.lifespan.adapters import ADAPTERS
from src.benchmarks.lifespan.landmark import cut_at_landmark
from src.features.static import static
from src.features.dynamics import dynamics
from src.benchmarks.lifespan.evaluate import run_evaluation_pipeline

def fit_and_score(X_train, Y_T_train, Y_E_train, X_test, Y_T_test, Y_E_test, n_components=10, seed=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_comp = min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0]-1)
    if n_comp < 1:
        return np.nan
        
    pca = PCA(n_components=n_comp, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
    df_train['T'] = Y_T_train
    df_train['E'] = Y_E_train
    
    df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
    
    cph = CoxPHFitter(penalizer=0.1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df_train, duration_col='T', event_col='E')
        preds = cph.predict_partial_hazard(df_test).values
        c_idx = concordance_index(Y_T_test, -preds, Y_E_test)
        return c_idx
    except Exception:
        return np.nan

def run_bootstrap(X, Y_T, Y_E, n_draws=400, seed=42, n_components=10, shuffle_y=False):
    c_indices = []
    n = len(X)
    rng = np.random.RandomState(seed)
    for i in range(n_draws):
        idx = rng.choice(n, size=n, replace=True)
        in_bag = np.unique(idx)
        out_of_bag = np.setdiff1d(np.arange(n), in_bag)
        
        if len(out_of_bag) == 0:
            continue
            
        X_train = X[idx]
        Y_T_train = Y_T[idx]
        Y_E_train = Y_E[idx]
        
        if shuffle_y:
            shuff_idx = rng.permutation(len(idx))
            Y_T_train = Y_T_train[shuff_idx]
            Y_E_train = Y_E_train[shuff_idx]
        
        X_test = X[out_of_bag]
        Y_T_test = Y_T[out_of_bag]
        Y_E_test = Y_E[out_of_bag]
        
        if shuffle_y:
            shuff_idx_test = rng.permutation(len(out_of_bag))
            Y_T_test = Y_T_test[shuff_idx_test]
            Y_E_test = Y_E_test[shuff_idx_test]
        
        c = fit_and_score(X_train, Y_T_train, Y_E_train, X_test, Y_T_test, Y_E_test, n_components=n_components, seed=seed+i)
        if not np.isnan(c):
            c_indices.append(c)
            
    return np.array(c_indices)

def evaluate_feature_set(X, Y_T, Y_E, n_components=10, seed=42):
    if len(X) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    c_obs_arr, _, _ = run_evaluation_pipeline(X, Y_T, Y_E, repeats=20, n_splits=5, seed=seed, n_components=n_components, shuffle_y=False)
    c_null_arr, _, _ = run_evaluation_pipeline(X, Y_T, Y_E, repeats=500, n_splits=5, seed=seed, n_components=n_components, shuffle_y=True)
    
    if len(c_obs_arr) == 0 or len(c_null_arr) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    c_obs = np.mean(c_obs_arr)
    null_mean = np.mean(c_null_arr)
    
    if not (0.47 <= null_mean <= 0.53):
        raise ValueError(f"Sanity check failed: Null mean {null_mean:.3f} is outside [0.47, 0.53]")
        
    p_value = np.mean(c_null_arr >= c_obs)
    detect_limit = np.percentile(c_null_arr, 95)
    
    c_boot = run_bootstrap(X, Y_T, Y_E, n_draws=400, seed=seed, n_components=n_components, shuffle_y=False)
    if len(c_boot) > 0:
        c_lo = np.percentile(c_boot, 2.5)
        c_hi = np.percentile(c_boot, 97.5)
    else:
        c_lo, c_hi = np.nan, np.nan
        
    return c_obs, c_lo, c_hi, null_mean, p_value, detect_limit

def get_features(animals, series, variables, L, min_valid_days):
    pre_series, outcomes = cut_at_landmark(animals, series, variables, L, min_valid_days)
    
    X_static_list = []
    X_dyn_list = []
    valid_animal_ids = outcomes['animal_id'].values
    
    for aid in valid_animal_ids:
        rows = pre_series[pre_series['animal_id'] == aid]
        f_stat = static(rows, variables)
        X_static_list.append(f_stat.values)
        
        f_dyn = dynamics(rows, variables)
        X_dyn_list.append(f_dyn.values)
        
    X_f1 = np.array(X_static_list) if X_static_list else np.empty((0,0))
    X_f2 = np.array(X_dyn_list) if X_dyn_list else np.empty((0,0))
    if X_f1.shape[0] > 0 and X_f2.shape[0] > 0:
        X_both = np.hstack([X_f1, X_f2])
    else:
        X_both = np.empty((0,0))
        
    Y_T = outcomes['T'].values
    Y_E = outcomes['E'].values
    return X_f1, X_f2, X_both, Y_T, Y_E, valid_animal_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, help="Path to the YAML configuration file defining the benchmark suite.")
    parser.add_argument('--dataset', required=True, help="Name of the dataset to run the sweep on.")
    args = parser.parse_args()

    with open(args.suite, 'r') as f:
        cfg = yaml.safe_load(f)

    datasets = cfg.get('datasets', [])
    ds = next((d for d in datasets if d['name'] == args.dataset), None)
    if not ds:
        raise ValueError(f"Dataset {args.dataset} not found in suite config.")
        
    min_valid_days = cfg.get('min_valid_days', 14)
    n_comp = cfg.get('model', {}).get('pca_components', 10)
    seed = cfg.get('cv', {}).get('seed', 42)
    
    adapter = ADAPTERS[args.dataset]
    animals, series, variables = adapter(ds)
    
    landmarks = [50, 60, 70, 80, 90, 100, 110]
    
    out_dir = Path(f"output/benchmarks/lifespan/{args.dataset}_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"{'L':>3} | {'FSet':>5} | {'N':>3} | {'Ev':>3} | {'C_obs':>5} | {'Boot_CI(95%)':>13} | {'Null':>5} | {'p-val':>5} | {'Limit':>5} | Signal")
    print("-" * 95)
    
    for L in landmarks:
        X_f1, X_f2, X_both, Y_T, Y_E, valid_animal_ids = get_features(animals, series, variables, L, min_valid_days)
        n_animals = len(valid_animal_ids)
        n_events = int(np.sum(Y_E))
        
        for fset, X in [('F1', X_f1), ('F2', X_f2), ('F1+F2', X_both)]:
            if X.shape[0] < 10:
                continue
                
            c_obs, c_lo, c_hi, null_mean, p_value, detect_limit = evaluate_feature_set(X, Y_T, Y_E, n_components=n_comp, seed=seed)
            if np.isnan(c_obs):
                continue
                
            signal = "SIGNAL" if p_value < 0.05 else "NO SIGNAL"
            
            print(f"{L:>3} | {fset:>5} | {n_animals:>3} | {n_events:>3} | {c_obs:.3f} | {c_lo:.3f}-{c_hi:.3f} | {null_mean:.3f} | {p_value:.3f} | {detect_limit:.3f} | {signal}")
            
            results.append({
                'L': L,
                'feature_set': fset,
                'n': n_animals,
                'events': n_events,
                'C': c_obs,
                'boot_lo': c_lo,
                'boot_hi': c_hi,
                'null_mean': null_mean,
                'p_value': p_value,
                'detect_limit': detect_limit,
                'signal': signal
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "landmark_sweep.csv", index=False)
    
if __name__ == '__main__':
    main()
