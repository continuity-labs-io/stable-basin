import os
import json
import pathlib
import pandas as pd
import numpy as np
import random
import argparse
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test
from scipy.fft import rfft, irfft, next_fast_len
import matplotlib.pyplot as plt
import warnings

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_f2a_for_fish(f_daily):
    X = f_daily.values
    t = f_daily.index.get_level_values('age_days').values
    mean_X = np.mean(X, axis=0)
    var_X = np.var(X, axis=0)
    
    t_center = t - np.mean(t)
    var_t = np.sum(t_center ** 2)
    
    if len(X) > 1 and var_t > 0:
        cov = np.sum(t_center[:, None] * (X - mean_X), axis=0)
        slope = cov / var_t
        X_detrended = X - mean_X - slope * t_center[:, None]
        
        n = X_detrended.shape[0]
        N = next_fast_len(2 * n)
        F = rfft(X_detrended, n=N, axis=0)
        acf = irfft(F * np.conj(F), n=N, axis=0)[:n]
        
        var_xf = np.sum(X_detrended ** 2, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            acf_norm = acf / var_xf
            acf_norm[:, var_xf == 0] = 0
            
        lag1_acf = acf_norm[1] if n > 1 else np.zeros(X.shape[1])
        
        iact = np.zeros(X.shape[1])
        max_lags = min(30, n)
        for f_idx in range(X.shape[1]):
            if var_xf[f_idx] == 0:
                continue
            acf_sum = 0
            for k in range(1, max_lags):
                if acf_norm[k, f_idx] <= 0:
                    break
                acf_sum += acf_norm[k, f_idx]
            iact[f_idx] = 1 + 2 * acf_sum
    else:
        lag1_acf = np.zeros(X.shape[1])
        iact = np.zeros(X.shape[1])
        
    return np.concatenate([var_X, lag1_acf, iact])

def run_evaluation_pipeline(X, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10, shuffle_y=False):
    X = np.array(X)
    Y_T = np.array(Y_T)
    Y_E = np.array(Y_E)
    
    c_indices = []
    oof_preds_rep0 = np.zeros(len(Y_T))
    failures = 0
    
    for r in range(repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof_preds = np.zeros(len(Y_T))
        
        if shuffle_y:
            np.random.seed(seed + r)
            idx = np.random.permutation(len(Y_T))
            Y_T_curr = Y_T[idx]
            Y_E_curr = Y_E[idx]
        else:
            Y_T_curr = Y_T
            Y_E_curr = Y_E
            
        repeat_failed = False
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_T_train, Y_T_test = Y_T_curr[train_index], Y_T_curr[test_index]
            Y_E_train, Y_E_test = Y_E_curr[train_index], Y_E_curr[test_index]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            n_comp = min(n_components, X_train.shape[1], X_train.shape[0]-1)
            if n_comp < 1:
                repeat_failed = True
                break
                
            pca = PCA(n_components=n_comp, random_state=seed+r)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
            df_train['T'] = Y_T_train
            df_train['E'] = Y_E_train
            
            df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
            
            cph = CoxPHFitter(penalizer=0.1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cph.fit(df_train, duration_col='T', event_col='E')
                preds = cph.predict_partial_hazard(df_test)
                oof_preds[test_index] = preds.values
            except Exception:
                repeat_failed = True
                break
                
        if repeat_failed:
            failures += 1
            continue
            
        if r == 0:
            oof_preds_rep0 = oof_preds.copy()
            
        try:
            c_idx = concordance_index(Y_T_curr, -oof_preds, Y_E_curr)
            c_indices.append(c_idx)
        except Exception:
            c_indices.append(0.5)
        
    return np.array(c_indices), oof_preds_rep0, failures

def run_loco_pipeline(X, Y_T, Y_E, cohorts, seed=42, n_components=10):
    unique_cohorts = np.unique(cohorts)
    if len(unique_cohorts) < 2:
        return 0.5, 0
    
    oof_preds = np.zeros(len(Y_T))
    oof_preds[:] = np.nan
    failures = 0
    
    for test_cohort in unique_cohorts:
        test_index = np.where(cohorts == test_cohort)[0]
        train_index = np.where(cohorts != test_cohort)[0]
        if len(train_index) == 0:
            continue
            
        X_train, X_test = X[train_index], X[test_index]
        Y_T_train, Y_T_test = Y_T[train_index], Y_T[test_index]
        Y_E_train, Y_E_test = Y_E[train_index], Y_E[test_index]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        n_comp = min(n_components, X_train.shape[1], X_train.shape[0]-1)
        if n_comp < 1:
            failures += 1
            continue
            
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
        df_train['T'] = Y_T_train
        df_train['E'] = Y_E_train
        
        df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
        
        cph = CoxPHFitter(penalizer=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df_train, duration_col='T', event_col='E')
            preds = cph.predict_partial_hazard(df_test)
            oof_preds[test_index] = preds.values
        except Exception:
            failures += 1
            
    valid_idx = ~np.isnan(oof_preds)
    if valid_idx.sum() > 0:
        try:
            c_idx = concordance_index(Y_T[valid_idx], -oof_preds[valid_idx], Y_E[valid_idx])
        except Exception:
            c_idx = 0.5
    else:
        c_idx = 0.5
        
    return c_idx, failures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str, choices=['auto', 'kinematic', 'syllable'], default='auto')
    args = parser.parse_args()
    seed_everything(args.seed)

    output_dir = pathlib.Path("output/benchmarks/lifespan")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_file = output_dir / "killifish_audit.json"
    results_file = output_dir / "killifish_results.json"
    
    print("STEP 0: DATA AUDIT")
    kinematic_file = pathlib.Path("data/killifish/data/b0_20250415/df_reformat_b0_10_20250416.csv")
    syllable_file = pathlib.Path("data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119_join_edit.csv")
    syllable_file_fallback = pathlib.Path("data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119.csv")
    
    if args.source == "auto":
        if kinematic_file.exists():
            csv_path, source_type = kinematic_file, "kinematic"
        elif syllable_file.exists():
            csv_path, source_type = syllable_file, "syllable"
        elif syllable_file_fallback.exists():
            csv_path, source_type = syllable_file_fallback, "syllable"
        else:
            raise FileNotFoundError("No source files found for auto.")
    elif args.source == "kinematic":
        csv_path, source_type = kinematic_file, "kinematic"
    else:
        if syllable_file.exists():
            csv_path, source_type = syllable_file, "syllable"
        else:
            csv_path, source_type = syllable_file_fallback, "syllable"
            
    print(f"Source type: {source_type}")
    print(f"Source file: {csv_path}")
    
    df_head = pd.read_csv(csv_path, nrows=3)
    print("\nColumns:", df_head.columns.tolist())
    print("First 3 rows:")
    print(df_head)
    
    if source_type == "kinematic":
        feature_cols = ['snout_velocity', 'snout_acceleration', 'disp', 'body_length', 'count_snout', 'active', 'inactive', 'sleep']
    else:
        feature_cols = [c for c in df_head.columns if c.startswith('state_')]
        
    forbidden = {'prognosis', 'prognosis_fraction', 'lifespan', 'status', 'hatch_date', 'full_fish_name', 'fish_number', 'cohort', 'table', 'sex', 'feeding', 'genotype'}
    for c in feature_cols:
        if c in forbidden:
            raise ValueError(f"Forbidden column in features: {c}")
            
    meta_cols = ['datetime', 'full_fish_name', 'fish_number', 'age_days', 'lifespan', 'status', 'genotype', 'sex', 'feeding', 'cohort']
    needed_cols = [c for c in feature_cols + meta_cols if c in df_head.columns]
    
    cache_file = output_dir / f"killifish_10min_{source_type}.parquet"
    if not cache_file.exists():
        chunks = []
        for chunk in pd.read_csv(csv_path, usecols=needed_cols, chunksize=100000):
            if 'feature' in chunk.columns:
                idx_cols = [c for c in chunk.columns if c not in ['feature', 'value']]
                chunk = chunk.pivot(index=idx_cols, columns='feature', values='value').reset_index()
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(cache_file)
    else:
        df = pd.read_parquet(cache_file)
        
    mapping = df.groupby('fish_number')['full_fish_name'].nunique()
    is_1_to_1 = (mapping == 1).all()
    print(f"\nfish_number maps 1:1 to full_fish_name: {is_1_to_1}")
    
    fish_meta = df.drop_duplicates('full_fish_name')
    if 'sex' in fish_meta.columns and 'feeding' in fish_meta.columns:
        print("\nFish counts by sex x feeding:")
        print(fish_meta.groupby(['sex', 'feeding']).size())
        
    df_primary = df[(df['genotype'] == 'wt') & (df['sex'] == 'male') & (df['feeding'] == 'al')].copy()
    fish_meta_primary = df_primary.drop_duplicates('full_fish_name')
    print("\nCounts by status (primary cohort):")
    print(fish_meta_primary['status'].value_counts())
    
    df_primary['is_valid_bin'] = df_primary[feature_cols].notna().all(axis=1)
    daily_valid_counts = df_primary.groupby(['full_fish_name', 'age_days'])['is_valid_bin'].sum()
    valid_days = daily_valid_counts[daily_valid_counts >= 120].reset_index()
    
    audit_data = {}
    max_count = -1
    selected_L = None
    for L in [70, 100]:
        count = 0
        for f in fish_meta_primary['full_fish_name'].unique():
            f_life = fish_meta_primary.loc[fish_meta_primary['full_fish_name'] == f, 'lifespan'].values[0]
            if f_life >= L:
                f_days = valid_days[(valid_days['full_fish_name'] == f) & (valid_days['age_days'] < L)]
                if len(f_days) >= 14:
                    count += 1
        if count >= 40 and count > max_count:
            max_count = count
            selected_L = L
        elif count >= 40 and count == max_count and L == 70:
            selected_L = L
            
    if selected_L is None:
        for L in [50, 60]:
            count = 0
            for f in fish_meta_primary['full_fish_name'].unique():
                f_life = fish_meta_primary.loc[fish_meta_primary['full_fish_name'] == f, 'lifespan'].values[0]
                if f_life >= L:
                    f_days = valid_days[(valid_days['full_fish_name'] == f) & (valid_days['age_days'] < L)]
                    if len(f_days) >= 14:
                        count += 1
            audit_data[f'L={L}'] = count
        with open(audit_file, "w") as f:
            json.dump(audit_data, f, indent=4)
        print(f"\nGATE 0 FAILED. Counts for 50/60: {audit_data}")
        return
        
    print(f"\nGATE 0 PASSED with L={selected_L} (count={max_count}). Proceeding to STEP 1.")
    audit_data['selected_L'] = selected_L
    audit_data['max_count'] = max_count
    with open(audit_file, "w") as f:
        json.dump(audit_data, f, indent=4)
        
    print("\nSTEP 1: FEATURE EXTRACTION")
    valid_fish = []
    for f in fish_meta_primary['full_fish_name'].unique():
        f_life = fish_meta_primary.loc[fish_meta_primary['full_fish_name'] == f, 'lifespan'].values[0]
        if f_life >= selected_L:
            f_days = valid_days[(valid_days['full_fish_name'] == f) & (valid_days['age_days'] < selected_L)]
            if len(f_days) >= 14:
                valid_fish.append(f)
                
    df_eval = df_primary[df_primary['full_fish_name'].isin(valid_fish)].copy()
    df_eval = df_eval.merge(valid_days[['full_fish_name', 'age_days']], on=['full_fish_name', 'age_days'], how='inner')
    
    daily_means = df_eval.groupby(['full_fish_name', 'age_days'])[feature_cols].mean()
    
    df_eval['datetime'] = pd.to_datetime(df_eval['datetime'])
    df_eval['minute_of_day'] = df_eval['datetime'].dt.hour * 60 + df_eval['datetime'].dt.minute
    
    f1_features = []
    f2a_features = []
    f2b_features = []
    
    for f in valid_fish:
        f_daily = daily_means.loc[f]
        f1_mean = f_daily.mean(axis=0).values
        f1_sd = f_daily.std(axis=0).values
        f1_features.append(np.concatenate([f1_mean, f1_sd]))
        
        f2a = compute_f2a_for_fish(f_daily)
        f2a_features.append(f2a)
        
        f_data = df_eval[df_eval['full_fish_name'] == f]
        profile_24h = f_data.groupby('minute_of_day')[feature_cols].mean()
        f_data = f_data.merge(profile_24h, on='minute_of_day', suffixes=('', '_mean'))
        
        resid_cols = []
        for col in feature_cols:
            r_col = f'{col}_resid'
            f_data[r_col] = f_data[col] - f_data[f'{col}_mean']
            resid_cols.append(r_col)
            
        daily_lag1 = []
        daily_ages = []
        for age, day_data in f_data.groupby('age_days'):
            day_data = day_data.sort_values('minute_of_day')
            X_resid = day_data[resid_cols].values
            
            if len(X_resid) > 1:
                var_r = np.var(X_resid, axis=0)
                cov_r = np.mean((X_resid[:-1] - np.mean(X_resid, axis=0)) * (X_resid[1:] - np.mean(X_resid, axis=0)), axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    lag1 = np.where(var_r > 0, cov_r / var_r, 0)
            else:
                lag1 = np.zeros(len(feature_cols))
                
            daily_lag1.append(lag1)
            daily_ages.append(age)
            
        daily_lag1 = np.array(daily_lag1)
        daily_ages = np.array(daily_ages)
        
        mean_lag1 = np.mean(daily_lag1, axis=0)
        t_center = daily_ages - np.mean(daily_ages)
        var_t = np.sum(t_center ** 2)
        if var_t > 0 and len(daily_ages) > 1:
            slope_lag1 = np.sum(t_center[:, None] * (daily_lag1 - mean_lag1), axis=0) / var_t
        else:
            slope_lag1 = np.zeros(len(feature_cols))
            
        f2b_features.append(np.concatenate([mean_lag1, slope_lag1]))
        
    X_f1 = np.array(f1_features)
    X_f2a = np.array(f2a_features)
    X_f2b = np.array(f2b_features)
    X_f2 = np.hstack([X_f2a, X_f2b])
    X_both = np.hstack([X_f1, X_f2])
    
    print(f"Feature counts: F1={X_f1.shape[1]}, F2={X_f2.shape[1]}, F1+F2={X_both.shape[1]}")
    
    print("\nSTEP 2: EVALUATION")
    Y_T = np.array([fish_meta_primary.loc[fish_meta_primary['full_fish_name'] == f, 'lifespan'].values[0] - selected_L for f in valid_fish])
    Y_E = np.array([(1 if fish_meta_primary.loc[fish_meta_primary['full_fish_name'] == f, 'status'].values[0] == 'd' else 0) for f in valid_fish])
    
    if 'cohort' in fish_meta_primary.columns:
        cohorts = np.array([fish_meta_primary.loc[fish_meta_primary['full_fish_name'] == f, 'cohort'].values[0] for f in valid_fish])
    else:
        cohorts = np.array([f.split('_')[-1] if 'cohort' in f else 'unknown' for f in valid_fish])
        
    c_f1, _, fails_f1 = run_evaluation_pipeline(X_f1, Y_T, Y_E, seed=args.seed)
    c_f2, _, fails_f2 = run_evaluation_pipeline(X_f2, Y_T, Y_E, seed=args.seed)
    c_both, oof_both_rep0, fails_both = run_evaluation_pipeline(X_both, Y_T, Y_E, seed=args.seed)
    c_null, _, fails_null = run_evaluation_pipeline(X_both, Y_T, Y_E, seed=args.seed, shuffle_y=True)
    
    loco_f1, loco_fails_f1 = run_loco_pipeline(X_f1, Y_T, Y_E, cohorts, seed=args.seed)
    loco_both, loco_fails_both = run_loco_pipeline(X_both, Y_T, Y_E, cohorts, seed=args.seed)
    
    print(f"Cox failures: F1={fails_f1}, F2={fails_f2}, F1+F2={fails_both}, Null={fails_null}")
    print(f"LOCO C-index F1: {loco_f1:.3f} (failures: {loco_fails_f1}), F1+F2: {loco_both:.3f} (failures: {loco_fails_both})")
    
    if len(c_both) == 0:
        print("All evaluations failed. Aborting.")
        return
        
    delta_c = c_both - c_f1 if len(c_both) == len(c_f1) and len(c_both) > 0 else np.array([0])
    
    gate1_pass = np.percentile(c_both, 2.5) > 0.5 if len(c_both) > 0 else False
    gate2_pass = (np.mean(delta_c) >= 0.03 and np.percentile(delta_c, 2.5) > 0) if len(delta_c) > 0 else False
    gate3_pass = (0.45 <= np.mean(c_null) <= 0.55) if len(c_null) > 0 else False
    
    print(f"G1: F1+F2 2.5th percentile > 0.5: {'PASS' if gate1_pass else 'FAIL'}")
    print(f"G2: delta C mean >= 0.03 and 2.5th percentile > 0: {'PASS' if gate2_pass else 'FAIL'}")
    print(f"G3: null mean in [0.45, 0.55]: {'PASS' if gate3_pass else 'FAIL'}")
    
    print("\nSTEP 3: OUTPUTS")
    results = {
        'source': source_type,
        'landmark': selected_L,
        'n_fish': len(valid_fish),
        'n_events': int(np.sum(Y_E)),
        'feature_counts': {'F1': X_f1.shape[1], 'F2': X_f2.shape[1], 'F1+F2': X_both.shape[1]},
        'failures': {'F1': fails_f1, 'F2': fails_f2, 'F1+F2': fails_both, 'Null': fails_null},
        'loco_c_index': {'F1': loco_f1, 'F1+F2': loco_both},
        'c_f1': {'mean': np.mean(c_f1) if len(c_f1)>0 else 0, '2.5%': np.percentile(c_f1, 2.5) if len(c_f1)>0 else 0, '97.5%': np.percentile(c_f1, 97.5) if len(c_f1)>0 else 0},
        'c_f2': {'mean': np.mean(c_f2) if len(c_f2)>0 else 0, '2.5%': np.percentile(c_f2, 2.5) if len(c_f2)>0 else 0, '97.5%': np.percentile(c_f2, 97.5) if len(c_f2)>0 else 0},
        'c_f1_f2': {'mean': np.mean(c_both) if len(c_both)>0 else 0, '2.5%': np.percentile(c_both, 2.5) if len(c_both)>0 else 0, '97.5%': np.percentile(c_both, 97.5) if len(c_both)>0 else 0},
        'c_null': {'mean': np.mean(c_null) if len(c_null)>0 else 0, '2.5%': np.percentile(c_null, 2.5) if len(c_null)>0 else 0, '97.5%': np.percentile(c_null, 97.5) if len(c_null)>0 else 0},
        'delta_c': {'mean': np.mean(delta_c) if len(delta_c)>0 else 0, '2.5%': np.percentile(delta_c, 2.5) if len(delta_c)>0 else 0, '97.5%': np.percentile(delta_c, 97.5) if len(delta_c)>0 else 0},
        'gates': {'G1': bool(gate1_pass), 'G2': bool(gate2_pass), 'G3': bool(gate3_pass)}
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    csv_file = pathlib.Path("results/lifespan_benchmark.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    base_row = {
        'date': date_str, 'dataset': 'killifish', 'source': source_type, 'landmark': selected_L,
        'n_fish': len(valid_fish), 'n_events': int(np.sum(Y_E)),
        'gate_G1': bool(gate1_pass), 'gate_G2': bool(gate2_pass), 'gate_G3': bool(gate3_pass)
    }
    
    rows = []
    if len(c_f1) > 0:
        r_f1 = base_row.copy(); r_f1.update({'feature_set': 'F1', 'c_mean': np.mean(c_f1), 'c_lo': np.percentile(c_f1, 2.5), 'c_hi': np.percentile(c_f1, 97.5), 'delta_c': ''})
        rows.append(r_f1)
    if len(c_f2) > 0:
        r_f2 = base_row.copy(); r_f2.update({'feature_set': 'F2', 'c_mean': np.mean(c_f2), 'c_lo': np.percentile(c_f2, 2.5), 'c_hi': np.percentile(c_f2, 97.5), 'delta_c': ''})
        rows.append(r_f2)
    if len(c_both) > 0:
        r_both = base_row.copy(); r_both.update({'feature_set': 'F1+F2', 'c_mean': np.mean(c_both), 'c_lo': np.percentile(c_both, 2.5), 'c_hi': np.percentile(c_both, 97.5), 'delta_c': np.mean(delta_c)})
        rows.append(r_both)
        
    if rows:
        df_res = pd.DataFrame(rows)
        if csv_file.exists():
            df_res.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_res.to_csv(csv_file, index=False)
            
    if len(c_both) > 0:
        plt.figure(figsize=(8, 6))
        means = [np.mean(c_f1), np.mean(c_f2), np.mean(c_both), np.mean(c_null)]
        los = [np.percentile(c_f1, 2.5), np.percentile(c_f2, 2.5), np.percentile(c_both, 2.5), np.percentile(c_null, 2.5)]
        his = [np.percentile(c_f1, 97.5), np.percentile(c_f2, 97.5), np.percentile(c_both, 97.5), np.percentile(c_null, 97.5)]
        errs = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
        
        plt.bar(['F1', 'F2', 'F1+F2', 'Null'], means, yerr=errs, capsize=5)
        plt.axhline(0.5, color='r', linestyle='--')
        plt.ylabel('C-index')
        plt.title(f'Performance by Feature Set ({source_type})')
        plt.savefig(output_dir / "killifish_cindex.png")
        plt.close()
        
        plt.figure(figsize=(8, 6))
        tertiles = pd.qcut(oof_both_rep0, 3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
        kmf = KaplanMeierFitter()
        
        for label in ['Low Risk', 'Medium Risk', 'High Risk']:
            mask = tertiles == label
            if np.any(mask):
                kmf.fit(Y_T[mask], event_observed=Y_E[mask], label=label)
                kmf.plot_survival_function()
                
        try:
            res = multivariate_logrank_test(Y_T, tertiles, Y_E)
            p_val = res.p_value
        except Exception:
            p_val = 1.0
            
        plt.title(f'Kaplan-Meier by Risk Tertile (log-rank p={p_val:.2e}, {source_type})')
        plt.xlabel('Days after Landmark')
        plt.ylabel('Survival Probability')
        plt.savefig(output_dir / "killifish_km.png")
        plt.close()

if __name__ == "__main__":
    main()
