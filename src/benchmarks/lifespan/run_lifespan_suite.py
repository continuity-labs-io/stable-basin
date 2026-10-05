"""
Lifespan Benchmark Runner

This script executes the core lifespan benchmark suite. It processes a configured list
of datasets (as defined in a YAML suite file), applies a specific landmarking strategy to 
measure static and dynamic behavioral features, and evaluates how well those features 
predict remaining lifespan (using the C-index). It rigorously validates the pipeline by 
checking both positive and negative controls, generating performance metrics, null distributions, 
and Kaplan-Meier survival plots for each dataset, and finally recording all results to a central ledger.
"""
import argparse
import yaml
import json
import subprocess
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from src.benchmarks.lifespan.adapters import ADAPTERS
from src.benchmarks.lifespan.contract import validate
from src.benchmarks.lifespan.landmark import cut_at_landmark
from src.features.static import static
from src.features.dynamics import dynamics
from src.benchmarks.lifespan.evaluate import run_evaluation_pipeline, run_loco_pipeline
from src.benchmarks.lifespan.controls import check_controls

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return 'unknown'

def evaluate_gate1(c_both):
    return bool(np.percentile(c_both, 2.5) > 0.5) if len(c_both) > 0 else False

def evaluate_gate2(delta_c):
    return bool(np.mean(delta_c) >= 0.03 and np.percentile(delta_c, 2.5) > 0) if len(delta_c) > 0 else False

def evaluate_gate3(c_null):
    return bool(0.45 <= np.mean(c_null) <= 0.55) if len(c_null) > 0 else False

def score_landmark(animals, series, variables, L, cfg, cut_fn=cut_at_landmark):
    min_valid_days = cfg.get('min_valid_days', 14)
    pre_series, outcomes = cut_fn(animals, series, variables, L, min_valid_days)
    
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
    cohorts = outcomes['group'].values

    eval_kwargs = {
        'repeats': cfg.get('cv', {}).get('repeats', 20),
        'n_splits': cfg.get('cv', {}).get('folds', 5),
        'seed': cfg.get('cv', {}).get('seed', 42),
        'n_components': cfg.get('model', {}).get('pca_components', 10)
    }

    results = {}
    
    if X_f1.shape[0] > 0:
        c_f1, _, fail_f1 = run_evaluation_pipeline(X_f1, Y_T, Y_E, **eval_kwargs)
        results['F1'] = (c_f1, fail_f1)
    else:
        results['F1'] = (np.array([]), 0)
        
    if X_f2.shape[0] > 0:
        c_f2, _, fail_f2 = run_evaluation_pipeline(X_f2, Y_T, Y_E, **eval_kwargs)
        results['F2'] = (c_f2, fail_f2)
    else:
        results['F2'] = (np.array([]), 0)
        
    if X_both.shape[0] > 0:
        c_both, oof_both, fail_both = run_evaluation_pipeline(X_both, Y_T, Y_E, **eval_kwargs)
        c_null, _, fail_null = run_evaluation_pipeline(X_both, Y_T, Y_E, shuffle_y=True, **eval_kwargs)
        results['F1+F2'] = (c_both, fail_both)
        results['null'] = (c_null, fail_null)
        results['oof_both'] = oof_both
        results['X_both'] = X_both
    else:
        results['F1+F2'] = (np.array([]), 0)
        results['null'] = (np.array([]), 0)
        results['oof_both'] = None
        results['X_both'] = np.empty((0,0))
        
    results['valid_animal_ids'] = valid_animal_ids
    results['Y_E'] = Y_E
    results['Y_T'] = Y_T
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, help="Path to the YAML configuration file defining the benchmark suite.")
    parser.add_argument('--only', help="Optional name of a single dataset to run. Controls run regardless.")
    args = parser.parse_args()

    with open(args.suite, 'r') as f:
        cfg = yaml.safe_load(f)
        
    forbidden_variables = set(cfg.get('forbidden_variables', []))

    datasets = cfg.get('datasets', [])
    if args.only:
        ds_controls = [ds for ds in datasets if ds.get('role', '') in ['positive_control', 'negative_control']]
        ds_only = [ds for ds in datasets if ds.get('name') == args.only and ds not in ds_controls]
        datasets = ds_controls + ds_only

    # Check for exactly one positive and one negative control
    pos_controls = [ds for ds in datasets if ds.get('role') == 'positive_control']
    neg_controls = [ds for ds in datasets if ds.get('role') == 'negative_control']
    overall_valid = True
    
    if len(pos_controls) != 1 or len(neg_controls) != 1:
        overall_valid = False

    ledger_path = Path('output/lifespan/lifespan_ledger.csv')
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_rows = []

    git_commit = get_git_commit()
    date_str = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    run_id = f"{date_str}-{git_commit}"
    
    control_results = {}
    control_failures = []
    
    # We will print these at the end
    print_lines = []

    for ds in datasets:
        ds_name = ds['name']
        ds_role = ds.get('role', 'data')
        
        adapter = ADAPTERS[ds_name]
        animals, series, variables = adapter(ds)
        
        audit = {
            'animals_by_group': animals['group'].value_counts().to_dict(),
            'animals_by_died': animals['died'].value_counts().to_dict(),
            'negative_value_counts': (series[variables] < 0).sum().to_dict(),
            'variables': variables,
            'valid_bin_rule': 'all_present'
        }
        
        validate(animals, series, variables, forbidden_variables)

        for L in ds.get('landmarks', []):
            res = score_landmark(animals, series, variables, L, {**cfg, **ds})
            valid_animal_ids = res['valid_animal_ids']
            Y_E = res['Y_E']
            Y_T = res['Y_T']
            
            c_f1, fail_f1 = res['F1']
            c_f2, fail_f2 = res['F2']
            c_both, fail_both = res['F1+F2']
            c_null, fail_null = res['null']
            oof_both = res.get('oof_both')
            X_both = res.get('X_both')
            
            # Check control conditions
            if ds_role in ['positive_control', 'negative_control']:
                if len(valid_animal_ids) == 0:
                    overall_valid = False
                
                if len(c_f1) == 0 or len(c_f2) == 0 or len(c_both) == 0:
                    overall_valid = False
                    
                if fail_f1 > 0 or fail_f2 > 0 or fail_both > 0:
                    overall_valid = False
            
            if ds_role == 'positive_control' and len(c_f2) > 0:
                control_results['positive_F2'] = np.mean(c_f2)
                print_lines.append(f"positive F2 C = {np.mean(c_f2):.3f}")
                
            if ds_role == 'negative_control':
                if len(c_f1) > 0:
                    control_results['negative_F1'] = np.mean(c_f1)
                if len(c_f2) > 0:
                    control_results['negative_F2'] = np.mean(c_f2)
                if len(c_both) > 0:
                    control_results['negative_F1+F2'] = np.mean(c_both)
                if len(c_f1) > 0 and len(c_f2) > 0 and len(c_both) > 0:
                    print_lines.append(f"negative F1 / F2 / F1+F2 C = {np.mean(c_f1):.3f} / {np.mean(c_f2):.3f} / {np.mean(c_both):.3f}")

            delta_c = c_both - c_f1 if len(c_both) == len(c_f1) and len(c_both) > 0 else np.array([])
            
            G1 = evaluate_gate1(c_both)
            G2 = evaluate_gate2(delta_c)
            G3 = evaluate_gate3(c_null)

            out_dir = Path(f"output/lifespan/{ds_name}/L{L}/{run_id}")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(out_dir / 'config.yaml', 'w') as f:
                yaml.dump(cfg, f)
            with open(out_dir / 'git_commit.txt', 'w') as f:
                f.write(git_commit)
            with open(out_dir / 'audit.json', 'w') as f:
                json.dump(audit, f, indent=4)
                
            if len(valid_animal_ids) > 0 and X_both.shape[0] > 0:
                df_features = pd.DataFrame({'animal_id': valid_animal_ids})
                idx_stat = [f"{v}_mean" for v in variables] + [f"{v}_sd" for v in variables]
                f2a_cols = [f"{v}_var_X" for v in variables] + [f"{v}_lag1_acf" for v in variables] + [f"{v}_iact" for v in variables]
                f2b_cols = [f"{v}_mean_lag1" for v in variables] + [f"{v}_slope_lag1" for v in variables]
                idx_both = idx_stat + f2a_cols + f2b_cols
                
                features_df_values = pd.DataFrame(X_both, columns=idx_both)
                df_features = pd.concat([df_features, features_df_values], axis=1)
                df_features.to_parquet(out_dir / 'features.parquet')

            results_json = {
                'c_f1': float(np.mean(c_f1)) if len(c_f1)>0 else 0,
                'c_f2': float(np.mean(c_f2)) if len(c_f2)>0 else 0,
                'c_both': float(np.mean(c_both)) if len(c_both)>0 else 0,
                'c_null': float(np.mean(c_null)) if len(c_null)>0 else 0,
            }
            with open(out_dir / 'results.json', 'w') as f:
                json.dump(results_json, f, indent=4)
                
            if len(c_both) > 0:
                plt.figure()
                plt.bar(['F1', 'F2', 'Both', 'Null'], [np.mean(c_f1), np.mean(c_f2), np.mean(c_both), np.mean(c_null)])
                plt.savefig(out_dir / 'cindex.png')
                plt.close()
                
                plt.figure()
                try:
                    tertiles = pd.qcut(oof_both, 3, labels=['Low', 'Med', 'High'])
                    kmf = KaplanMeierFitter()
                    for label in ['Low', 'Med', 'High']:
                        mask = tertiles == label
                        if np.any(mask):
                            kmf.fit(Y_T[mask], event_observed=Y_E[mask], label=label)
                            kmf.plot_survival_function()
                except Exception:
                    pass
                plt.savefig(out_dir / 'km.png')
                plt.close()

            def append_row(fset, c_array):
                if len(c_array) > 0:
                    ledger_rows.append({
                        'run_id': run_id,
                        'date': date_str,
                        'git_commit': git_commit,
                        'dataset': ds_name,
                        'role': ds_role,
                        'landmark': L,
                        'feature_set': fset,
                        'n_animals': len(valid_animal_ids),
                        'n_events': int(np.sum(Y_E)),
                        'c_mean': np.mean(c_array),
                        'c_lo': np.percentile(c_array, 2.5),
                        'c_hi': np.percentile(c_array, 97.5),
                        'G1': G1,
                        'G2': G2,
                        'G3': G3,
                        'valid': None
                    })
                    
            append_row('F1', c_f1)
            append_row('F2', c_f2)
            append_row('F1+F2', c_both)
            
    failures = check_controls(control_results)
    if failures:
        overall_valid = False
        for f in failures:
            print(f)

    for row in ledger_rows:
        row['valid'] = overall_valid
        
    if ledger_rows:
        df_ledger = pd.DataFrame(ledger_rows)
        cols = ['run_id', 'date', 'git_commit', 'dataset', 'role', 'landmark', 'feature_set', 'n_animals', 'n_events', 'c_mean', 'c_lo', 'c_hi', 'G1', 'G2', 'G3', 'valid']
        df_ledger = df_ledger[cols]
        if ledger_path.exists():
            df_ledger.to_csv(ledger_path, mode='a', header=False, index=False)
        else:
            df_ledger.to_csv(ledger_path, index=False)

    for pl in print_lines:
        print(pl)
    if not overall_valid:
        print("FAIL")
        import sys
        sys.exit(1)
    else:
        print("PASS")

if __name__ == '__main__':
    main()
