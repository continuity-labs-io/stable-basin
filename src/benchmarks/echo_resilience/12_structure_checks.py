"""
12_structure_checks.py

Checks if the model has the structure the code claims.
"""

import argparse
import os
import json
import logging
import yaml
import numpy as np

import jax
import jax.numpy as jnp
from sklearn.linear_model import Ridge

from src.data.utils import zscore_fit, window_starts
import importlib
null_control = importlib.import_module("src.benchmarks.echo_resilience.11_null_control")
load_ts = null_control.load_ts
load_or_train_graph = null_control.load_or_train_graph

from src.echo.metrics.energy_landscape import ScalarEnergy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def integrated_autocorrelation_time(x):
    """Integrated autocorrelation time up to first zero crossing."""
    N = x.shape[0]
    mu = np.mean(x)
    v = x - mu
    var = np.mean(v**2)
    if var == 0:
        return np.nan
    acf = np.correlate(v, v, mode='full')[N - 1:] / (N * var)
    tau = 0.0
    for i in range(len(acf)):
        if acf[i] <= 0:
            break
        tau += acf[i]
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/echo_resilience.yaml")
    ap.add_argument("--train-ts", default="data/worm/EigenWorms_TRAIN.ts")
    ap.add_argument("--test-ts", default="data/worm/EigenWorms_TEST.ts")
    ap.add_argument("--weights", default="")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out-dir", default="output/benchmarks/echo_resilience")
    ap.add_argument("--windows-per-worm", type=int, default=4)
    ap.add_argument("--burn-in", type=int, default=50)
    args = ap.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if "seq_len" not in config.get("dataset", {}):
        config.setdefault("dataset", {})["seq_len"] = config["dataset"].get("ebm_seq_len", 100)
        
    if args.retrain:
        if "optimization" not in config:
            config["optimization"] = {}
        config["optimization"]["max_epochs"] = 1
        
    dataset_name = config.get("dataset", {}).get("name", "worm_gait")
    if not args.weights:
        args.weights = config.get("paths", {}).get("model_weights", f"output/benchmarks/echo_resilience/{dataset_name}/{dataset_name}_trained_engine.eqx")
        
    train_trajs, train_labels = load_ts(args.train_ts)
    test_trajs, test_labels = load_ts(args.test_ts)
    
    mu, sd = zscore_fit(train_trajs)
    test_trajs = [(t - mu) / (sd + 1e-8) for t in test_trajs]
    
    # 20 held-out TEST worms
    seed = config.get("experiment", {}).get("seed", 42)
    rng = np.random.default_rng(seed)
    
    # Choose 20 worms from test
    n_worms_total = len(test_trajs)
    keep = np.sort(rng.choice(n_worms_total, min(20, n_worms_total), replace=False))
    test_trajs = [test_trajs[i] for i in keep]
    
    graph = load_or_train_graph(config, args.config, train_trajs, train_labels, args)
    
    # Run forced unroll
    seq_len = config.get("dataset", {}).get("seq_len", 500)
    dt = config.get("experiment", {}).get("dt", 0.01)
    d_state = graph.d_micro + graph.d_macro
    d_micro = graph.d_micro
    d_macro_full = graph.d_macro

    import equinox as eqx
    @eqx.filter_jit
    def unroll(g, x_inits, seqs, keys):
        return jax.vmap(lambda xi, s, k: g.forced_unroll(k, xi, dt, seq=s))(x_inits, seqs, keys)

    worm_trajs = [] # shape: (20 worms, steps, d_state)
    worm_trajs_windowed = []
    worm_unroll_args = []
    for w, traj in enumerate(test_trajs):
        starts = window_starts(traj.shape[0], seq_len, args.windows_per_worm)
        seqs = np.stack([traj[s : s + seq_len] for s in starts]).astype(np.float32)
        rngs = [np.random.default_rng([seed, w, win]) for win in range(len(starts))]
        x_inits = np.stack([0.01 * r.standard_normal(d_state) for r in rngs]).astype(np.float32)
        base = jax.random.PRNGKey(seed)
        keys = jax.numpy.stack([jax.random.fold_in(base, w * 10_000 + win) for win in range(len(starts))])
        
        states = unroll(graph, x_inits, seqs, keys) # shape (windows, seq_len, d_state)
        X_win = np.array(states)[:, args.burn_in :, :]
        worm_trajs_windowed.append(X_win)
        X = X_win.reshape(-1, d_state)
        worm_trajs.append(X)
        worm_unroll_args.append({"x_inits": x_inits, "seqs": seqs, "keys": keys})
        
    worm_trajs = np.array(worm_trajs)
    
    # ---------------------------------------------------------
    # TEST A: Markov blanket in the learned energy
    # ---------------------------------------------------------
    logger.info("TEST A: Markov blanket in the learned energy")
    hull = graph.flow_factor.micro_hull
    d_int = hull.d_internal
    d_sen = hull.d_sensory
    d_act = hull.d_active
    d_ext = hull.d_external
    
    internal = np.arange(0, d_int)
    sensory = np.arange(d_int, d_int + d_sen)
    active = np.arange(d_int + d_sen, d_int + d_sen + d_act)
    external = np.arange(d_int + d_sen + d_act, d_int + d_sen + d_act + d_ext)
    
    energy_fn = ScalarEnergy(graph.ebm)
    
    @eqx.filter_jit
    def micro_hessian_block(x):
        H = jax.hessian(energy_fn)(x)
        return H[:d_micro, :d_micro]

    flat_states = worm_trajs.reshape(-1, d_state)
    pool_indices = rng.choice(flat_states.shape[0], min(500, flat_states.shape[0]), replace=False)
    pool_states = flat_states[pool_indices]
    
    r_vals = []
    for x in pool_states:
        H_u = micro_hessian_block(jnp.array(x))
        H_ie = H_u[jnp.ix_(internal, external)]
        norm_ie = jnp.linalg.norm(H_ie, ord='fro')
        norm_Hu = jnp.linalg.norm(H_u, ord='fro')
        r_vals.append(float(norm_ie / (norm_Hu + 1e-8)))
        
    r_vals = np.array(r_vals)
    r_median = np.median(r_vals)
    r_p95 = np.percentile(r_vals, 95)
    
    pass_A = bool(r_p95 < 0.01)
    logger.info(f"Test A: median(r)={r_median:.4f}, p95(r)={r_p95:.4f}. PASS: {pass_A}")
    
    # ---------------------------------------------------------
    # TEST B: is the macro level slower than the micro level?
    # ---------------------------------------------------------
    logger.info("TEST B: is the macro level slower than the micro level?")
    
    d_macro_int = graph.flow_factor.macro_hull.d_internal
    micro_int_idx = internal
    macro_int_idx = np.arange(d_micro, d_micro + d_macro_int)
    
    ratios = []
    for w in range(len(worm_trajs)):
        x = worm_trajs[w]
        tau_micro = [integrated_autocorrelation_time(x[:, i]) for i in micro_int_idx]
        tau_macro = [integrated_autocorrelation_time(x[:, i]) for i in macro_int_idx]
        
        median_tau_micro = np.nanmedian(tau_micro)
        median_tau_macro = np.nanmedian(tau_macro)
        ratios.append(median_tau_macro / (median_tau_micro + 1e-8))
        
    ratios = np.array(ratios)
    
    n_boot = 5000
    boot_ratios = []
    for _ in range(n_boot):
        sample = rng.choice(ratios, len(ratios), replace=True)
        boot_ratios.append(np.median(sample))
    
    ci_lower = np.percentile(boot_ratios, 2.5)
    ci_upper = np.percentile(boot_ratios, 97.5)
    med_ratio = np.median(ratios)
    pass_B = bool(ci_lower > 2.0)
    logger.info(f"Test B: ratio={med_ratio:.2f} CI95[{ci_lower:.2f}, {ci_upper:.2f}]. PASS: {pass_B}")

    # ---------------------------------------------------------
    # TEST C: does the macro level predict itself?
    # ---------------------------------------------------------
    logger.info("TEST C: does the macro level predict itself?")
    
    def extract_features_targets_windowed(X_win, lag=5):
        target_traj = X_win[:, :, d_micro : d_micro + d_macro_int]
        macro_traj = X_win[:, :, d_micro : d_micro + d_macro_full]
        
        X_macro = []
        X_full = []
        Y = []
        
        windows, T, _ = X_win.shape
        for win in range(windows):
            for t in range(lag - 1, T - 1):
                X_macro.append(macro_traj[win, t-lag+1 : t+1].flatten())
                X_full.append(X_win[win, t-lag+1 : t+1].flatten())
                Y.append(target_traj[win, t+1])
                
        return np.array(X_macro), np.array(X_full), np.array(Y)

    X_m_list, X_f_list, Y_list = [], [], []
    groups_list = []
    for w in range(len(worm_trajs_windowed)):
        xm, xf, y = extract_features_targets_windowed(worm_trajs_windowed[w])
        X_m_list.append(xm)
        X_f_list.append(xf)
        Y_list.append(y)
        groups_list.append(np.full(len(y), w))

    train_idx = np.arange(10)
    test_idx = np.arange(10, min(20, len(worm_trajs_windowed)))

    X_m_train = np.concatenate([X_m_list[i] for i in train_idx])
    X_f_train = np.concatenate([X_f_list[i] for i in train_idx])
    Y_train = np.concatenate([Y_list[i] for i in train_idx])
    groups_train = np.concatenate([groups_list[i] for i in train_idx])
    
    m_mean = X_m_train.mean(axis=0)
    m_std = X_m_train.std(axis=0) + 1e-8
    f_mean = X_f_train.mean(axis=0)
    f_std = X_f_train.std(axis=0) + 1e-8
    
    X_m_train = (X_m_train - m_mean) / m_std
    X_f_train = (X_f_train - f_mean) / f_std
    
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    
    alphas = np.logspace(-3, 3, 13)
    cv_macro = list(GroupKFold(n_splits=5).split(X_m_train, Y_train, groups_train))
    cv_full = list(GroupKFold(n_splits=5).split(X_f_train, Y_train, groups_train))
    
    model_macro = RidgeCV(alphas=alphas, cv=cv_macro)
    model_full = RidgeCV(alphas=alphas, cv=cv_full)
    
    model_macro.fit(X_m_train, Y_train)
    model_full.fit(X_f_train, Y_train)
    
    preds_macro_train = model_macro.predict(X_m_train)
    preds_full_train = model_full.predict(X_f_train)
    mse_macro_train = np.mean((Y_train - preds_macro_train)**2)
    mse_full_train = np.mean((Y_train - preds_full_train)**2)
    
    logger.info(f"Test C: train MSE_macro={mse_macro_train:.4f}, train MSE_full={mse_full_train:.4f}")
    if mse_full_train > mse_macro_train + 1e-6:
        logger.error("Sanity check failed: train MSE_full > train MSE_macro. The fitting code is wrong.")
        raise ValueError("Sanity check failed: train MSE_full > train MSE_macro")
        
    test_mse_macro_list = []
    test_mse_full_list = []
    
    for i in test_idx:
        X_m_t = (X_m_list[i] - m_mean) / m_std
        X_f_t = (X_f_list[i] - f_mean) / f_std
        Y_t = Y_list[i]
        
        p_m = model_macro.predict(X_m_t)
        p_f = model_full.predict(X_f_t)
        
        test_mse_macro_list.append(np.mean((Y_t - p_m)**2))
        test_mse_full_list.append(np.mean((Y_t - p_f)**2))
        
    test_mse_macro_list = np.array(test_mse_macro_list)
    test_mse_full_list = np.array(test_mse_full_list)
    
    mse_macro_test = np.mean(test_mse_macro_list)
    mse_full_test = np.mean(test_mse_full_list)
    closure_point = mse_full_test / (mse_macro_test + 1e-8)
    
    n_boot_c = 5000
    boot_closures = []
    for _ in range(n_boot_c):
        idx_sample = rng.choice(len(test_mse_macro_list), len(test_mse_macro_list), replace=True)
        sample_macro = np.mean(test_mse_macro_list[idx_sample])
        sample_full = np.mean(test_mse_full_list[idx_sample])
        boot_closures.append(sample_full / (sample_macro + 1e-8))
        
    ci_closure_lower = np.percentile(boot_closures, 2.5)
    ci_closure_upper = np.percentile(boot_closures, 97.5)
    
    decoupling_changes = []
    for idx_pos, i in enumerate(test_idx):
        args_i = worm_unroll_args[i]
        other_pos = (idx_pos + 1) % len(test_idx)
        other_i = test_idx[other_pos]
        other_seqs = worm_unroll_args[other_i]["seqs"]
        
        alt_states = unroll(graph, args_i["x_inits"], other_seqs, args_i["keys"])
        alt_X_win = np.array(alt_states)[:, args.burn_in :, :]
        
        orig_X_win = worm_trajs_windowed[i]
        
        orig_macro = orig_X_win[:, :, d_micro : d_micro + d_macro_int]
        alt_macro = alt_X_win[:, :, d_micro : d_micro + d_macro_int]
        
        change = np.mean(np.abs(alt_macro - orig_macro))
        orig_sd = orig_macro.std() + 1e-8
        decoupling_changes.append(change / orig_sd)
        
    mean_decoupling = np.mean(decoupling_changes)
    
    pass_C = False
    fail_C = False
    
    if closure_point >= 0.9 and ci_closure_upper <= 1.1 and mean_decoupling >= 0.1:
        pass_C = True
    elif closure_point < 0.9:
        fail_C = True
        
    status_C = "PASS" if pass_C else ("FAIL" if fail_C else "VOID")
    logger.info(f"Test C: closure={closure_point:.4f} CI95[{ci_closure_lower:.4f}, {ci_closure_upper:.4f}] "
                f"decoupling={mean_decoupling:.4f}. Status: {status_C}")

    import hashlib
    def get_sha256(filepath):
        if not os.path.exists(filepath):
            return None
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
            
    files_loaded = {
        "train_ts": {"path": args.train_ts, "sha256": get_sha256(args.train_ts)},
        "test_ts": {"path": args.test_ts, "sha256": get_sha256(args.test_ts)},
        "weights": {"path": args.weights, "sha256": get_sha256(args.weights)}
    }
    
    results = {
        "files_loaded": files_loaded,
        "test_a": {
            "median_r": float(r_median),
            "p95_r": float(r_p95),
            "pass": pass_A
        },
        "test_b": {
            "median_ratio": float(med_ratio),
            "ci95_lower": float(ci_lower),
            "ci95_upper": float(ci_upper),
            "pass": pass_B
        },
        "test_c": {
            "closure_median": float(closure_point),
            "ci95_lower": float(ci_closure_lower),
            "ci95_upper": float(ci_closure_upper),
            "mean_decoupling_change": float(mean_decoupling),
            "train_mse_macro": float(mse_macro_train),
            "train_mse_full": float(mse_full_train),
            "test_mse_macro": float(mse_macro_test),
            "test_mse_full": float(mse_full_test),
            "status": status_C
        }
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "structure_checks_c_v2.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_file}")

if __name__ == "__main__":
    main()
