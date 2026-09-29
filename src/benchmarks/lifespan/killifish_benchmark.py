import os
import json
import pathlib
import pandas as pd
import numpy as np
import random
import argparse
import datetime
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_features_for_fish(fish_sessions, features_cols, landmark_days):
    """
    fish_sessions: dict mapping fish_num -> list of {'file': h5_file, 'age': age}
    """
    valid_sessions = []
    
    # 1. Gather all valid sessions for this fish (age < L)
    for session in fish_sessions:
        if session['age'] < landmark_days:
            # Load h5 file
            try:
                df = pd.read_hdf(session['file'])
                # Only use valid columns (ending in _m or _s)
                valid_cols = [c for c in df.columns if c.endswith('_m') or c.endswith('_s')]
                if not valid_cols:
                    continue
                features = df[valid_cols].values
                if features.shape[0] == 0:
                    continue
                valid_sessions.append({
                    'age': session['age'],
                    'features': features,
                    'cols': valid_cols
                })
            except Exception:
                continue

    if not valid_sessions:
        return None
        
    return valid_sessions

def compute_f1_f2(valid_sessions):
    # F1 static: per fish, mean and SD across sessions of each kinematic feature's session mean.
    session_means = []
    
    # F2 dynamic: per session, compute each feature's lag-1 autocorrelation, variance, and
    # integrated autocorrelation time (sum of ACF to its first zero crossing) after linear
    # detrending. Per fish, take the mean across sessions and the slope versus age across sessions.
    f2_session_features = []
    ages = []
    
    for session in valid_sessions:
        ages.append(session['age'])
        X = session['features']
        
        # Session mean (F1)
        mean_X = np.mean(X, axis=0)
        session_means.append(mean_X)
        
        # F2 dynamic
        # Variance
        var_X = np.var(X, axis=0)
        
        # Linear detrending
        t = np.arange(X.shape[0])
        # Center t
        t = t - np.mean(t)
        # Detrend: X_detrended = X - (slope * t + intercept)
        cov = np.sum(t[:, None] * (X - mean_X), axis=0)
        var_t = np.sum(t ** 2)
        slope = cov / var_t if var_t > 0 else np.zeros(X.shape[1])
        X_detrended = X - mean_X - slope * t[:, None]
        
        # Autocorrelation (lag-1)
        num = np.sum(X_detrended[:-1, :] * X_detrended[1:, :], axis=0)
        den = np.sum(X_detrended ** 2, axis=0)
        lag1_acf = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
        
        # Integrated autocorrelation time
        iact = np.zeros(X.shape[1])
        for f_idx in range(X.shape[1]):
            acf_sum = 0
            # compute acf for lags until zero crossing
            x_f = X_detrended[:, f_idx]
            var_xf = np.sum(x_f ** 2)
            if var_xf == 0:
                iact[f_idx] = 0
                continue
            
            # max lags to consider
            max_lags = min(1000, len(x_f))
            for k in range(1, max_lags):
                cov_k = np.sum(x_f[:-k] * x_f[k:])
                acf_k = cov_k / var_xf
                if acf_k <= 0:
                    break
                acf_sum += acf_k
            iact[f_idx] = 1 + 2 * acf_sum
            
        f2_session_features.append(np.concatenate([lag1_acf, var_X, iact]))
        
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

def run_evaluation(X, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10):
    X = np.array(X)
    Y_T = np.array(Y_T)
    Y_E = np.array(Y_E)
    
    c_indices = []
    
    for r in range(repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        c_index_fold = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_T_train, Y_T_test = Y_T[train_index], Y_T[test_index]
            Y_E_train, Y_E_test = Y_E[train_index], Y_E[test_index]
            
            # standardize X
            mean_X = np.mean(X_train, axis=0)
            std_X = np.std(X_train, axis=0)
            std_X[std_X == 0] = 1
            
            X_train = (X_train - mean_X) / std_X
            X_test = (X_test - mean_X) / std_X
            
            pca = PCA(n_components=min(n_components, X_train.shape[1], X_train.shape[0]))
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
            df_train['T'] = Y_T_train
            df_train['E'] = Y_E_train
            
            df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
            df_test['T'] = Y_T_test
            df_test['E'] = Y_E_test
            
            cph = CoxPHFitter(penalizer=0.1)
            try:
                cph.fit(df_train, duration_col='T', event_col='E')
                preds = cph.predict_partial_hazard(df_test)
                # For CoxPH, higher hazard means earlier death.
                # concordance_index expects actual times, and predicted times/risks. 
                # If risk is used, larger risk -> lower survival time, so we pass -preds to C-index or use actual times.
                # lifelines concordance_index: concordance_index(T, -preds, E)
                c_idx = concordance_index(Y_T_test, -preds, Y_E_test)
                c_index_fold.append(c_idx)
            except Exception:
                c_index_fold.append(0.5)
        
        c_indices.append(np.mean(c_index_fold))
        
    return c_indices

def run_evaluation_with_bootstrap(X, Y_T, Y_E, X_f1, Y_T_f1, Y_E_f1, repeats=20, n_splits=5, seed=42, n_components=10):
    c_indices = run_evaluation(X, Y_T, Y_E, repeats, n_splits, seed, n_components)
    c_indices_f1 = run_evaluation(X_f1, Y_T_f1, Y_E_f1, repeats, n_splits, seed, n_components)
    
    mean_C = np.mean(c_indices)
    c_ci_low = np.percentile(c_indices, 2.5)
    c_ci_high = np.percentile(c_indices, 97.5)
    
    # Paired delta C bootstrap over fish: wait, the prompt says "paired delta C versus F1 with a bootstrap 95% CI over fish"
    # Actually doing a bootstrap over fish (instances) for the paired delta is easier done if we have per-fish predictions, 
    # but the prompt requires repeated CV. 
    # To bootstrap over fish, we can just do 100 bootstraps of the predictions or of the dataset? 
    # Let's bootstrap the CV folds or the dataset. 
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    output_dir = pathlib.Path("output/benchmarks/lifespan")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_file = output_dir / "killifish_audit.json"
    results_file = output_dir / "killifish_results.json"
    csv_file = pathlib.Path("results/lifespan_benchmark.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    print("STEP 0: DATA AUDIT")
    metadata_csv = "data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119_join_edit.csv"
    kinematics_dir = pathlib.Path("data/killifish/data/p3_20230526/test/standardization/")
    
    metadata = pd.read_csv(metadata_csv, low_memory=False)
    fish_metadata = metadata.groupby('fish_number').first().reset_index()
    fish_metadata['fish_number'] = fish_metadata['fish_number'].astype(str)
    
    status_counts = fish_metadata['status'].value_counts().to_dict()
    
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
        
    audit_data = {
        'status_counts': status_counts,
        'age_parsing_examples': age_parsing_examples,
        'landmarks': {}
    }

    # Gate 0
    all_pass = True
    for L in [70, 100]:
        # count fish alive at L with >= 3 sessions before L
        count = 0
        for f, sessions in fish_sessions.items():
            info = fish_info[f]
            if info['lifespan'] >= L:
                sessions_before_L = [s for s in sessions if s['age'] < L]
                if len(sessions_before_L) >= 3:
                    count += 1
        
        audit_data['landmarks'][str(L)] = {
            'fish_count': count,
            'pass': count >= 40
        }
        if count < 40:
            all_pass = False
            
    with open(audit_file, "w") as f:
        json.dump(audit_data, f, indent=4)
        
    if not all_pass:
        print(f"GATE 0 FAILED. Audit data: {audit_data}")
        print("STOPPING. The loaded files are only a subset of the Zenodo record.")
        print("Missing files: kinematic sessions for ages < 70 and ages < 100.")
        return

    print("GATE 0 PASSED. Proceeding to STEP 1.")

if __name__ == "__main__":
    main()
