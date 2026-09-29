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
null_control = importlib.import_module("src.benchmarks.aging_resilience.11_null_control")
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
    ap.add_argument("--config", default="configs/aging_resilience.yaml")
    ap.add_argument("--train-ts", default="data/worm/EigenWorms_TRAIN.ts")
    ap.add_argument("--test-ts", default="data/worm/EigenWorms_TEST.ts")
    ap.add_argument("--weights", default="output/benchmarks/aging_resilience/05_worm_gait_decline_trained_engine.eqx")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out-dir", default="output/benchmarks/aging_resilience")
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
    if args.weights == "output/benchmarks/aging_resilience/05_worm_gait_decline_trained_engine.eqx":
        args.weights = config.get("paths", {}).get("model_weights", f"output/benchmarks/aging_resilience/{dataset_name}_trained_engine.eqx")
        
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
    for w, traj in enumerate(test_trajs):
        starts = window_starts(traj.shape[0], seq_len, args.windows_per_worm)
        seqs = np.stack([traj[s : s + seq_len] for s in starts]).astype(np.float32)
        rngs = [np.random.default_rng([seed, w, win]) for win in range(len(starts))]
        x_inits = np.stack([0.01 * r.standard_normal(d_state) for r in rngs]).astype(np.float32)
        base = jax.random.PRNGKey(seed)
        keys = jax.numpy.stack([jax.random.fold_in(base, w * 10_000 + win) for win in range(len(starts))])
        
        states = unroll(graph, x_inits, seqs, keys) # shape (windows, seq_len, d_state)
        X = np.array(states)[:, args.burn_in :, :].reshape(-1, d_state)
        worm_trajs.append(X)
        
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
    
    def extract_features_targets(traj, lag=5):
        target_traj = traj[:, d_micro : d_micro + d_macro_int]
        macro_traj = traj[:, d_micro : d_micro + d_macro_full]
        
        X_macro = []
        X_full = []
        Y = []
        
        for t in range(lag - 1, traj.shape[0] - 1):
            X_macro.append(macro_traj[t-lag+1 : t+1].flatten())
            X_full.append(traj[t-lag+1 : t+1].flatten())
            Y.append(target_traj[t+1])
            
        return np.array(X_macro), np.array(X_full), np.array(Y)

    X_m_list, X_f_list, Y_list = [], [], []
    for w in range(len(worm_trajs)):
        xm, xf, y = extract_features_targets(worm_trajs[w])
        X_m_list.append(xm)
        X_f_list.append(xf)
        Y_list.append(y)

    train_idx = np.arange(10)
    test_idx = np.arange(10, min(20, len(worm_trajs)))

    X_m_train = np.concatenate([X_m_list[i] for i in train_idx])
    X_f_train = np.concatenate([X_f_list[i] for i in train_idx])
    Y_train = np.concatenate([Y_list[i] for i in train_idx])
    
    X_m_test = np.concatenate([X_m_list[i] for i in test_idx])
    X_f_test = np.concatenate([X_f_list[i] for i in test_idx])
    Y_test = np.concatenate([Y_list[i] for i in test_idx])

    model_macro = Ridge(alpha=1.0)
    model_full = Ridge(alpha=1.0)
    
    model_macro.fit(X_m_train, Y_train)
    model_full.fit(X_f_train, Y_train)
    
    preds_macro = model_macro.predict(X_m_test)
    preds_full = model_full.predict(X_f_test)
    
    mse_macro = np.mean((Y_test - preds_macro)**2)
    mse_full = np.mean((Y_test - preds_full)**2)
    
    closure = float(mse_full / (mse_macro + 1e-8))
    pass_C = bool(closure >= 0.9)
    logger.info(f"Test C: closure={closure:.4f} (MSE_full={mse_full:.4f}, MSE_macro={mse_macro:.4f}). PASS: {pass_C}")

    results = {
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
            "closure": float(closure),
            "mse_macro": float(mse_macro),
            "mse_full": float(mse_full),
            "pass": pass_C
        }
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "structure_checks.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_file}")

if __name__ == "__main__":
    main()
