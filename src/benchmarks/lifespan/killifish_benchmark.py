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

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_f1_f2_for_session(X):
    # F1 static: session mean
    mean_X = np.mean(X, axis=0)
    
    # F2 dynamic
    var_X = np.var(X, axis=0)
    
    # Linear detrending
    t = np.arange(X.shape[0])
    t = t - np.mean(t)
    var_t = np.sum(t ** 2)
    
    if X.shape[0] > 1 and var_t > 0:
        cov = np.sum(t[:, None] * (X - mean_X), axis=0)
        slope = cov / var_t
        X_detrended = X - mean_X - slope * t[:, None]
        
        # Autocorrelation using FFT
        n = X_detrended.shape[0]
        N = next_fast_len(2 * n)
        F = rfft(X_detrended, n=N, axis=0)
        acf = irfft(F * np.conj(F), n=N, axis=0)[:n]
        
        var_xf = np.sum(X_detrended ** 2, axis=0)
        # Handle zero variance
        with np.errstate(divide='ignore', invalid='ignore'):
            acf_norm = acf / var_xf
            acf_norm[:, var_xf == 0] = 0
            
        lag1_acf = acf_norm[1] if n > 1 else np.zeros(X.shape[1])
        
        # IACT
        iact = np.zeros(X.shape[1])
        max_lags = min(500, n)
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
        
    return np.concatenate([mean_X, lag1_acf, var_X, iact])

def load_or_compute_session_features(valid_sessions, cache_file):
    cache = {}
    if cache_file.exists():
        df_cache = pd.read_parquet(cache_file)
        cache = df_cache.set_index('file').to_dict('index')
    
    new_cache_rows = []
    for session in valid_sessions:
        fpath = session['file']
        if fpath in cache:
            session['cached_features'] = cache[fpath]['features_array']
        else:
            feats = compute_f1_f2_for_session(session['features'])
            session['cached_features'] = feats
            new_cache_rows.append({'file': fpath, 'features_array': feats})
            
    if new_cache_rows:
        df_new = pd.DataFrame(new_cache_rows)
        if cache_file.exists():
            df_cache_all = pd.concat([pd.read_parquet(cache_file), df_new], ignore_index=True)
        else:
            df_cache_all = df_new
        df_cache_all.to_parquet(cache_file)

def get_fish_features(valid_sessions):
    session_means = []
    f2_session_features = []
    ages = []
    
    # 4 metrics: mean, lag1_acf, var, iact. Each is length n_features
    # Thus cached_features length is 4 * n_features
    n_features = len(valid_sessions[0]['cached_features']) // 4
    
    for session in valid_sessions:
        ages.append(session['age'])
        sess_feats = session['cached_features']
        mean_X = sess_feats[:n_features]
        f2_feats = sess_feats[n_features:]
        
        session_means.append(mean_X)
        f2_session_features.append(f2_feats)
        
    session_means = np.array(session_means)
    f2_session_features = np.array(f2_session_features)
    ages = np.array(ages)
    
    f1_mean = np.mean(session_means, axis=0)
    f1_sd = np.std(session_means, axis=0)
    f1 = np.concatenate([f1_mean, f1_sd])
    
    f2_mean = np.mean(f2_session_features, axis=0)
    
    if len(ages) > 1:
        t = ages - np.mean(ages)
        var_t = np.sum(t ** 2)
        if var_t > 0:
            f2_slope = np.sum(t[:, None] * (f2_session_features - f2_mean), axis=0) / var_t
        else:
            f2_slope = np.zeros_like(f2_mean)
    else:
        f2_slope = np.zeros_like(f2_mean)
        
    f2 = np.concatenate([f2_mean, f2_slope])
    return f1, f2

def run_evaluation_pipeline(X, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10, shuffle_y=False):
    X = np.array(X)
    Y_T = np.array(Y_T)
    Y_E = np.array(Y_E)
    
    c_indices = []
    oof_preds_rep0 = np.zeros(len(Y_T))
    
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
            
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_T_train, Y_T_test = Y_T_curr[train_index], Y_T_curr[test_index]
            Y_E_train, Y_E_test = Y_E_curr[train_index], Y_E_curr[test_index]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            n_comp = min(n_components, X_train.shape[1], X_train.shape[0]-1)
            if n_comp < 1:
                oof_preds[test_index] = np.random.rand(len(test_index))
                continue
                
            pca = PCA(n_components=n_comp, random_state=seed+r)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
            df_train['T'] = Y_T_train
            df_train['E'] = Y_E_train
            
            df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
            
            cph = CoxPHFitter(penalizer=0.1)
            try:
                cph.fit(df_train, duration_col='T', event_col='E')
                preds = cph.predict_partial_hazard(df_test)
                oof_preds[test_index] = preds.values
            except Exception:
                oof_preds[test_index] = np.random.rand(len(test_index))
                
        if r == 0:
            oof_preds_rep0 = oof_preds.copy()
            
        try:
            c_idx = concordance_index(Y_T_curr, -oof_preds, Y_E_curr)
        except ZeroDivisionError:
            c_idx = 0.5
        c_indices.append(c_idx)
        
    return np.array(c_indices), oof_preds_rep0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kinematics-dir", type=str, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    output_dir = pathlib.Path("output/benchmarks/lifespan")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_file = output_dir / "killifish_audit.json"
    results_file = output_dir / "killifish_results.json"
    cache_file = output_dir / "session_features.parquet"
    csv_file = pathlib.Path("results/lifespan_benchmark.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    print("STEP 0: DATA AUDIT")
    metadata_csv = "data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119_join_edit.csv"
    kinematics_dir = pathlib.Path(args.kinematics_dir)
    
    metadata = pd.read_csv(metadata_csv, low_memory=False)
    fish_metadata = metadata.groupby('fish_number').first().reset_index()
    fish_metadata['fish_number'] = fish_metadata['fish_number'].astype(str)
    
    fish_info = {}
    for _, row in fish_metadata.iterrows():
        f = str(row['fish_number'])
        fish_info[f] = {
            'lifespan': float(row['lifespan']),
            'status': row['status'],
            'event': 1 if row['status'] == 'd' else 0,
        }
        
    fish_sessions = {}
    if kinematics_dir.exists():
        h5_files = list(kinematics_dir.rglob("*.h5"))
        h5_files.sort()
        for h5_file in h5_files:
            raw_fish_name = h5_file.parent.name
            if raw_fish_name.startswith("fish"):
                fish_num = raw_fish_name.split('_')[0].replace('fish', '')
            else:
                fish_num = raw_fish_name
                
            if fish_num in fish_info:
                try:
                    chronological_age = float(raw_fish_name.split('_')[1])
                except Exception:
                    continue
                    
                if fish_num not in fish_sessions:
                    fish_sessions[fish_num] = []
                fish_sessions[fish_num].append({
                    'file': str(h5_file),
                    'age': chronological_age
                })

    age_parsing_examples = []
    for f in list(fish_sessions.keys())[:5]:
        s = fish_sessions[f][0]
        age_parsing_examples.append({
            'path': s['file'],
            'parsed_age': s['age']
        })
    print("Parsed age examples:")
    for ex in age_parsing_examples:
        print(ex)

    audit_data = {
        'age_parsing_examples': age_parsing_examples,
        'landmarks': {},
        'files': {}
    }

    # Gate 0
    selected_L = None
    max_count = -1
    for L in [70, 100]:
        count = 0
        for f, sessions in fish_sessions.items():
            if fish_info[f]['lifespan'] >= L:
                if len([s for s in sessions if s['age'] < L]) >= 3:
                    count += 1
        audit_data['landmarks'][str(L)] = {'fish_count': count, 'pass': count >= 40}
        if count >= 40 and count > max_count:
            selected_L = L
            max_count = count
            
    if selected_L is None:
        for L in [50, 60]:
            count = 0
            for f, sessions in fish_sessions.items():
                if fish_info[f]['lifespan'] >= L:
                    if len([s for s in sessions if s['age'] < L]) >= 3:
                        count += 1
            audit_data['landmarks'][str(L)] = {'fish_count': count, 'pass': False}
        with open(audit_file, "w") as f:
            json.dump(audit_data, f, indent=4)
        print(f"GATE 0 FAILED. Counts for 50/60: {audit_data['landmarks']}")
        return

    print(f"GATE 0 PASSED with L={selected_L} (count={max_count}). Proceeding to STEP 1.")

    print("STEP 1: FEATURE EXTRACTION")
    fish_f1 = {}
    fish_f2 = {}
    
    # Process features
    for f, sessions in fish_sessions.items():
        if fish_info[f]['lifespan'] >= selected_L:
            valid_sessions = []
            for session in sessions:
                if session['age'] < selected_L:
                    try:
                        df = pd.read_hdf(session['file'])
                        valid_cols = [c for c in df.columns if c.endswith('_m') or c.endswith('_s')]
                        if not valid_cols: continue
                        features = df[valid_cols].values
                        n_rows = features.shape[0]
                        if n_rows == 0: continue
                        
                        if session['file'] not in audit_data['files']:
                            audit_data['files'][session['file']] = {'n_rows': n_rows, 'n_cols': len(valid_cols)}
                        
                        if n_rows > 50000:
                            block_size = int(np.ceil(n_rows / 50000))
                            audit_data['block_size_used'] = block_size
                            n_blocks = n_rows // block_size
                            features = features[:n_blocks * block_size]
                            features = features.reshape(n_blocks, block_size, -1).mean(axis=1)
                            
                        valid_sessions.append({'age': session['age'], 'features': features, 'file': session['file']})
                    except Exception:
                        continue
                        
            if len(valid_sessions) >= 3:
                load_or_compute_session_features(valid_sessions, cache_file)
                f1, f2 = get_fish_features(valid_sessions)
                fish_f1[f] = f1
                fish_f2[f] = f2

    with open(audit_file, "w") as f:
        json.dump(audit_data, f, indent=4)

    print("STEP 2: EVALUATION")
    valid_fish = list(fish_f1.keys())
    if not valid_fish:
        print("No valid fish after filtering.")
        return
        
    X_f1 = np.array([fish_f1[f] for f in valid_fish])
    X_f2 = np.array([fish_f2[f] for f in valid_fish])
    X_both = np.hstack([X_f1, X_f2])
    
    Y_T = np.array([fish_info[f]['lifespan'] - selected_L for f in valid_fish])
    Y_E = np.array([fish_info[f]['event'] for f in valid_fish])
    
    print("Evaluating F1...")
    c_f1, _ = run_evaluation_pipeline(X_f1, Y_T, Y_E, seed=args.seed)
    print("Evaluating F2...")
    c_f2, _ = run_evaluation_pipeline(X_f2, Y_T, Y_E, seed=args.seed)
    print("Evaluating F1+F2...")
    c_both, oof_both_rep0 = run_evaluation_pipeline(X_both, Y_T, Y_E, seed=args.seed)
    print("Evaluating Null...")
    c_null, _ = run_evaluation_pipeline(X_both, Y_T, Y_E, seed=args.seed, shuffle_y=True)
    
    delta_c = c_both - c_f1
    
    gate1_pass = np.percentile(c_both, 2.5) > 0.5
    gate2_pass = np.mean(delta_c) >= 0.03 and np.percentile(delta_c, 2.5) > 0
    gate3_pass = 0.45 <= np.mean(c_null) <= 0.55
    
    print(f"G1: F1+F2 2.5th percentile > 0.5: {'PASS' if gate1_pass else 'FAIL'}")
    print(f"G2: delta C mean >= 0.03 and 2.5th percentile > 0: {'PASS' if gate2_pass else 'FAIL'}")
    print(f"G3: null mean in [0.45, 0.55]: {'PASS' if gate3_pass else 'FAIL'}")

    print("STEP 3: OUTPUTS")
    results = {
        'landmark': selected_L,
        'n_fish': len(valid_fish),
        'n_events': int(np.sum(Y_E)),
        'feature_counts': {'F1': X_f1.shape[1], 'F2': X_f2.shape[1], 'F1+F2': X_both.shape[1]},
        'block_size_used': audit_data.get('block_size_used', 1),
        'c_f1': {'mean': np.mean(c_f1), '2.5%': np.percentile(c_f1, 2.5), '97.5%': np.percentile(c_f1, 97.5)},
        'c_f2': {'mean': np.mean(c_f2), '2.5%': np.percentile(c_f2, 2.5), '97.5%': np.percentile(c_f2, 97.5)},
        'c_f1_f2': {'mean': np.mean(c_both), '2.5%': np.percentile(c_both, 2.5), '97.5%': np.percentile(c_both, 97.5)},
        'c_null': {'mean': np.mean(c_null), '2.5%': np.percentile(c_null, 2.5), '97.5%': np.percentile(c_null, 97.5)},
        'delta_c': {'mean': np.mean(delta_c), '2.5%': np.percentile(delta_c, 2.5), '97.5%': np.percentile(delta_c, 97.5)},
        'gates': {'G1': bool(gate1_pass), 'G2': bool(gate2_pass), 'G3': bool(gate3_pass)}
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    rows = []
    base_row = {
        'date': date_str, 'dataset': 'killifish', 'landmark': selected_L,
        'n_fish': len(valid_fish), 'n_events': int(np.sum(Y_E)),
        'gate_G1': bool(gate1_pass), 'gate_G2': bool(gate2_pass), 'gate_G3': bool(gate3_pass)
    }
    
    r_f1 = base_row.copy(); r_f1.update({'feature_set': 'F1', 'c_mean': np.mean(c_f1), 'c_lo': np.percentile(c_f1, 2.5), 'c_hi': np.percentile(c_f1, 97.5), 'delta_c': ''})
    r_f2 = base_row.copy(); r_f2.update({'feature_set': 'F2', 'c_mean': np.mean(c_f2), 'c_lo': np.percentile(c_f2, 2.5), 'c_hi': np.percentile(c_f2, 97.5), 'delta_c': ''})
    r_both = base_row.copy(); r_both.update({'feature_set': 'F1+F2', 'c_mean': np.mean(c_both), 'c_lo': np.percentile(c_both, 2.5), 'c_hi': np.percentile(c_both, 97.5), 'delta_c': np.mean(delta_c)})
    
    df_res = pd.DataFrame([r_f1, r_f2, r_both])
    if csv_file.exists():
        df_res.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_res.to_csv(csv_file, index=False)
        
    # Plots
    plt.figure(figsize=(8, 6))
    means = [np.mean(c_f1), np.mean(c_f2), np.mean(c_both), np.mean(c_null)]
    los = [np.percentile(c_f1, 2.5), np.percentile(c_f2, 2.5), np.percentile(c_both, 2.5), np.percentile(c_null, 2.5)]
    his = [np.percentile(c_f1, 97.5), np.percentile(c_f2, 97.5), np.percentile(c_both, 97.5), np.percentile(c_null, 97.5)]
    errs = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
    
    plt.bar(['F1', 'F2', 'F1+F2', 'Null'], means, yerr=errs, capsize=5)
    plt.axhline(0.5, color='r', linestyle='--')
    plt.ylabel('C-index')
    plt.title('Performance by Feature Set')
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
        
    plt.title(f'Kaplan-Meier by Risk Tertile (log-rank p={p_val:.2e})')
    plt.xlabel('Days after Landmark')
    plt.ylabel('Survival Probability')
    plt.savefig(output_dir / "killifish_km.png")
    plt.close()

if __name__ == "__main__":
    main()
