"""
11_null_control.py  ->  suggested location: src/benchmarks/worm_gait/11_null_control.py

Worm-level null and synthetic positive control for the Hessian-trace pipeline.

Arms
  NULL      Held-out (TEST) worms, all clean, split into disjoint strain-stratified
            halves A and B. Any "effect" here is the pipeline's false-positive floor.
            Because per-worm summaries are computed once, the split is redrawn
            --n-splits times at no model cost, giving an empirical false-positive rate.
  POSITIVE  The same worms with gait-amplitude relaxation slowed by known factors s
            (synthetic_aging.slow_amplitude_relaxation). Scored unpaired (clean A vs
            degraded B, the realistic cohort design) and paired (each worm vs itself).
            s = 1.0 is the sham arm and must reproduce the clean data exactly.

Statistics are computed on one summary per worm (n = worms). The timestep-level KS
test used in the draft paper is also reported, labelled as pseudoreplicated, so its
inflation is visible next to the worm-level result. NaN traces are dropped and
counted, never imputed.

Pre-registered direction: if Hessian trace tracks loss of resilience ("flattening"),
degraded worms should show LOWER trace, i.e. negative Hedges' g (degraded minus clean).

Usage
  python -m src.benchmarks.worm_gait.11_null_control --config configs/worm_gait_ebm.yaml
  python -m src.benchmarks.worm_gait.11_null_control --retrain          # fresh model, held-out val
  python 11_null_control.py --stub --train-ts a.ts --test-ts b.ts       # stats plumbing only, no JAX
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

try:
    from src.data.behavior.synthetic_aging import amplitude_residual_stats, slow_amplitude_relaxation
except ImportError:  # running next to synthetic_aging.py outside the repo
    from synthetic_aging import amplitude_residual_stats, slow_amplitude_relaxation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================ data

def load_ts(path: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Parse a UEA .ts file, keeping the class label (the repo loader drops it)."""
    trajs, labels, in_data = [], [], False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split(":")
            dims = [[float(v) for v in p.split(",") if v] for p in parts[:-1]]
            trajs.append(np.asarray(dims, dtype=np.float64).T)  # (T, n_dims)
            labels.append(parts[-1].strip())
    if not trajs:
        raise ValueError(f"No series parsed from {path}.")
    return trajs, np.asarray(labels)


def zscore_fit(trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    allx = np.concatenate(trajs, axis=0)
    return allx.mean(axis=0), allx.std(axis=0, ddof=1)  # ddof=1 matches torch.std


def stratified_split(labels: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    idx_a, idx_b = [], []
    for lab in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == lab))
        cut = int(round(frac * len(idx)))
        idx_a.extend(idx[:cut])
        idx_b.extend(idx[cut:])
    return np.sort(idx_a), np.sort(idx_b)


def window_starts(T: int, seq_len: int, k: int) -> np.ndarray:
    if T < seq_len:
        raise ValueError(f"Series length {T} < seq_len {seq_len}.")
    return np.unique(np.linspace(0, T - seq_len, k).astype(int))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================ model

def load_or_train_graph(config: dict, config_path: str, train_trajs, train_labels, args):
    import equinox as eqx
    import jax
    from src.benchmarks.worm_gait.core import build_graph
    from src.echo.primitives.ebm import PrecisionWeightedEBM

    seed = config.get("experiment", {}).get("seed", 42)
    graph, d_state = build_graph(PrecisionWeightedEBM, jax.random.PRNGKey(seed), config)

    if not args.retrain:
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"{args.weights} not found. Run benchmark 05 or pass --retrain.")
        logger.info(f"Loading trained engine from {args.weights}")
        return eqx.tree_deserialise_leaves(args.weights, graph)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from src.data.datasets import JAXDictDataset
    from src.echo.harness.echo_runner import EchoRunner
    from src.echo.harness.echo_trainer import EchoTrainer

    seq_len = config["dataset"]["seq_len"]

    class CropDataset(Dataset):
        def __init__(self, trajs):
            self.trajs = [t for t in trajs if t.shape[0] >= seq_len]

        def __len__(self):
            return len(self.trajs)

        def __getitem__(self, i):
            t = self.trajs[i]
            s = random.randint(0, t.shape[0] - seq_len)
            return torch.tensor(t[s : s + seq_len], dtype=torch.float32)

    tr_idx, va_idx = stratified_split(train_labels, np.random.default_rng(seed), frac=0.85)
    bs = config.get("dataset", {}).get("batch_size", 2)
    train_loader = DataLoader(
        JAXDictDataset(CropDataset([train_trajs[i] for i in tr_idx]), d_state), batch_size=bs, shuffle=True
    )
    val_loader = DataLoader(
        JAXDictDataset(CropDataset([train_trajs[i] for i in va_idx]), d_state), batch_size=bs, shuffle=False
    )
    opt = config.get("optimization", {})
    trainer = EchoTrainer(graph, learning_rate=opt.get("learning_rate", 1e-4), max_grad_norm=opt.get("max_grad_norm", 0.1))
    runner = EchoRunner(config_path)
    runner.setup(trainer)
    logger.info(f"Training fresh engine: {len(tr_idx)} train worms, {len(va_idx)} held-out val worms")
    dt = config.get("experiment", {}).get("dt", 0.01)
    graph = runner.run(graph, train_loader, val_loader, jax.random.PRNGKey(seed + 1), dt=dt)
    os.makedirs(args.out_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(args.out_dir, "11_retrained_engine.eqx"), graph)
    return graph


class HessianTraceEvaluator:
    """Forced unroll over fixed windows, then exact Hessian trace of the joint energy.

    Common random numbers: x_init and PRNG keys depend only on (worm_id, window), so
    clean and degraded versions of the same worm see identical noise.
    """

    def __init__(self, graph, dt: float, seed: int, burn_in: int, trace_batch: int):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.graph, self.seed, self.burn_in, self.trace_batch = graph, seed, burn_in, trace_batch
        self.d_state = graph.d_micro + graph.d_macro
        d_micro = graph.d_micro

        @eqx.filter_jit
        def unroll(g, x_inits, seqs, keys):
            return jax.vmap(lambda xi, s, k: g.forced_unroll(k, xi, dt, seq=s))(x_inits, seqs, keys)

        @eqx.filter_jit
        def traces(ff, X):
            def energy(x):
                return ff.joint_energy_fn(x[:d_micro], x[d_micro:])

            return jax.vmap(lambda x: jnp.trace(jax.hessian(energy)(x)))(X)

        self._unroll, self._traces = unroll, traces

    def __call__(self, traj: np.ndarray, worm_id: int, starts: np.ndarray, seq_len: int):
        jax, jnp = self.jax, self.jnp
        seqs = np.stack([traj[s : s + seq_len] for s in starts]).astype(np.float32)
        rngs = [np.random.default_rng([self.seed, worm_id, w]) for w in range(len(starts))]
        x_inits = np.stack([0.01 * r.standard_normal(self.d_state) for r in rngs]).astype(np.float32)
        base = jax.random.PRNGKey(self.seed)
        keys = jnp.stack([jax.random.fold_in(base, worm_id * 10_000 + w) for w in range(len(starts))])

        states = self._unroll(self.graph, jnp.asarray(x_inits), jnp.asarray(seqs), keys)
        X = states[:, self.burn_in :, :].reshape(-1, self.d_state)
        out = []
        for i in range(0, X.shape[0], self.trace_batch):
            chunk = X[i : i + self.trace_batch]
            pad = self.trace_batch - chunk.shape[0]
            if pad:
                chunk = jnp.concatenate([chunk, jnp.repeat(chunk[-1:], pad, axis=0)], axis=0)
            out.append(np.asarray(self._traces(self.graph.flow_factor, chunk))[: self.trace_batch - pad])
        t = np.concatenate(out)
        finite = np.isfinite(t)
        return t[finite], int((~finite).sum())


class StubEvaluator:
    """NOT A MODEL. Deterministic function of the input, for testing the stats plumbing."""

    def __init__(self, burn_in: int):
        self.burn_in = burn_in

    def __call__(self, traj, worm_id, starts, seq_len):
        x = np.concatenate([traj[s + self.burn_in : s + seq_len] for s in starts])
        return 30.0 + 10.0 * np.tanh(x[:, 0] ** 2 + x[:, 1] ** 2 - 1.0), 0


# ============================================================================ statistics

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """(mean(b) - mean(a)) / pooled SD, small-sample corrected."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((b.mean() - a.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def unpaired_stats(a, b, rng, n_perm: int, n_boot: int) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    obs = b.mean() - a.mean()
    pooled = np.concatenate([a, b])
    perms = np.argsort(rng.random((n_perm, pooled.size)), axis=1)
    diffs = pooled[perms[:, len(a) :]].mean(1) - pooled[perms[:, : len(a)]].mean(1)
    boot = [hedges_g(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(obs),
        "perm_p": float((np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)),
        "mannwhitney_p": float(mannwhitneyu(a, b).pvalue),
        "hedges_g": hedges_g(a, b),
        "hedges_g_ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
    }


def paired_stats(clean, degraded, rng, n_perm: int, n_boot: int) -> dict:
    d = np.asarray(degraded) - np.asarray(clean)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(1))
    boot = [rng.choice(d, d.size).mean() for _ in range(n_boot)]
    sd = d.std(ddof=1)
    return {
        "n_worms": int(d.size),
        "mean_delta": float(d.mean()),
        "mean_delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "signflip_p": float((np.sum(null >= abs(d.mean())) + 1) / (n_perm + 1)),
    }


def naive_timestep_ks(steps_a: list[np.ndarray], steps_b: list[np.ndarray]) -> dict:
    a, b = np.concatenate(steps_a), np.concatenate(steps_b)
    ks = ks_2samp(a, b)
    return {"n_a": int(a.size), "n_b": int(b.size), "ks": float(ks.statistic), "p": float(ks.pvalue),
            "note": "pseudoreplicated: timesteps treated as independent; shown for comparison only"}


def resplit_stats(clean_s, cond_s, labels, n_splits, seed) -> dict:
    gs, ps = [], []
    for r in range(n_splits):
        ia, ib = stratified_split(labels, np.random.default_rng(seed + 1 + r))
        gs.append(hedges_g(clean_s[ia], cond_s[ib]))
        ps.append(mannwhitneyu(clean_s[ia], cond_s[ib]).pvalue)
    gs, ps = np.asarray(gs), np.asarray(ps)
    return {
        "n_splits": int(n_splits),
        "frac_p_lt_0.05": float(np.mean(ps < 0.05)),
        "g_mean": float(np.nanmean(gs)),
        "g_pct": {k: float(np.nanpercentile(gs, q)) for k, q in (("p2.5", 2.5), ("p50", 50), ("p97.5", 97.5))},
        "abs_g_p95": float(np.nanpercentile(np.abs(gs), 95)),
    }


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worm_gait_ebm.yaml")
    ap.add_argument("--train-ts", default="data/worm/EigenWorms_TRAIN.ts")
    ap.add_argument("--test-ts", default="data/worm/EigenWorms_TEST.ts")
    ap.add_argument("--weights", default="output/benchmarks/worm_gait/05_worm_gait_decline_trained_engine.eqx")
    ap.add_argument("--retrain", action="store_true", help="train a fresh engine with a held-out val split")
    ap.add_argument("--out-dir", default="output/benchmarks/worm_gait")
    ap.add_argument("--severities", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1], help="eigenworm channels forming the gait oscillator")
    ap.add_argument("--windows-per-worm", type=int, default=4)
    ap.add_argument("--burn-in", type=int, default=50, help="unroll steps discarded per window (x_init transient)")
    ap.add_argument("--summary", choices=["median", "mean"], default="median")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--trace-batch", type=int, default=2048)
    ap.add_argument("--max-worms", type=int, default=None, help="cap TEST worms for a quick run")
    ap.add_argument("--stub", action="store_true", help="stats plumbing test with a fake trace function; no JAX")
    args = ap.parse_args()

    if 1.0 not in args.severities:
        args.severities = [1.0] + list(args.severities)
    severities = sorted(set(args.severities))
    pair = tuple(args.pair)

    config = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    elif not args.stub:
        raise FileNotFoundError(args.config)
    seq_len = config.get("dataset", {}).get("seq_len", 500)
    dt = config.get("experiment", {}).get("dt", 0.01)
    seed = config.get("experiment", {}).get("seed", 42)
    if args.burn_in >= seq_len:
        raise ValueError("--burn-in must be smaller than seq_len.")

    # ---- data: normalise both files with TRAIN statistics
    train_trajs, train_labels = load_ts(args.train_ts)
    test_trajs, test_labels = load_ts(args.test_ts)
    mu, sd = zscore_fit(train_trajs)
    train_trajs = [(t - mu) / (sd + 1e-8) for t in train_trajs]
    test_trajs = [(t - mu) / (sd + 1e-8) for t in test_trajs]
    if args.max_worms:
        keep = np.sort(np.random.default_rng(seed).choice(len(test_trajs), args.max_worms, replace=False))
        test_trajs, test_labels = [test_trajs[i] for i in keep], test_labels[keep]
    n_worms = len(test_trajs)
    logger.info(f"TEST worms: {n_worms}; classes: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    # ---- model
    if args.stub:
        logger.warning("STUB MODE: trace values are a fake function of the input. Plumbing test only.")
        evaluate = StubEvaluator(args.burn_in)
    else:
        graph = load_or_train_graph(config, args.config, train_trajs, train_labels, args)
        evaluate = HessianTraceEvaluator(graph, dt, seed, args.burn_in, args.trace_batch)

    # ---- per-worm traces for every severity (s = 1.0 is clean / sham)
    agg = np.median if args.summary == "median" else np.mean
    summaries = {s: np.full(n_worms, np.nan) for s in severities}
    steps = {s: [None] * n_worms for s in severities}
    manip = {s: [] for s in severities}
    nan_counts = {s: 0 for s in severities}
    for s in severities:
        logger.info(f"Evaluating severity s = {s}")
        for w, traj in enumerate(test_trajs):
            x = traj if s == 1.0 else slow_amplitude_relaxation(traj, s, pair)
            manip[s].append(amplitude_residual_stats(x, pair))
            t, n_nan = evaluate(x, w, window_starts(x.shape[0], seq_len, args.windows_per_worm), seq_len)
            nan_counts[s] += n_nan
            steps[s][w] = t
            summaries[s][w] = agg(t) if t.size else np.nan

    for s in severities:
        bad = np.isnan(summaries[s])
        if bad.any():
            logger.warning(f"s={s}: {bad.sum()} worms had no finite traces and are excluded.")
    valid = ~np.any([np.isnan(summaries[s]) for s in severities], axis=0)
    labels_v = test_labels[valid]
    S = {s: summaries[s][valid] for s in severities}
    steps_v = {s: [steps[s][i] for i in np.flatnonzero(valid)] for s in severities}

    rng = np.random.default_rng(seed)
    ia, ib = stratified_split(labels_v, np.random.default_rng(args.split_seed))
    clean = S[1.0]

    results = {
        "provenance": {
            "train_ts": args.train_ts, "train_sha256": sha256(args.train_ts),
            "test_ts": args.test_ts, "test_sha256": sha256(args.test_ts),
            "weights": None if (args.stub or args.retrain) else args.weights,
            "stub": args.stub, "retrained": args.retrain, "config": config,
            "summary": args.summary, "windows_per_worm": args.windows_per_worm, "burn_in": args.burn_in,
            "pair": list(pair), "n_worms_used": int(valid.sum()), "nan_traces_dropped": nan_counts,
        },
        "null": {
            "primary_split": unpaired_stats(clean[ia], clean[ib], rng, args.n_perm, args.n_boot),
            "naive_timestep_ks": naive_timestep_ks([steps_v[1.0][i] for i in ia], [steps_v[1.0][i] for i in ib]),
            "resplits": resplit_stats(clean, clean, labels_v, args.n_splits, args.split_seed),
        },
        "positive": {},
    }

    for s in severities:
        m = manip[s]
        entry = {
            "manipulation_check": {
                "amp_ar1_median": float(np.median([d["amp_ar1"] for d in m])),
                "amp_var_median": float(np.median([d["amp_var"] for d in m])),
            }
        }
        if s != 1.0:
            entry["unpaired_primary"] = unpaired_stats(clean[ia], S[s][ib], rng, args.n_perm, args.n_boot)
            entry["naive_timestep_ks"] = naive_timestep_ks(
                [steps_v[1.0][i] for i in ia], [steps_v[s][i] for i in ib]
            )
            entry["unpaired_resplits"] = resplit_stats(clean, S[s], labels_v, args.n_splits, args.split_seed)
            entry["paired_all_worms"] = paired_stats(clean, S[s], rng, args.n_perm, args.n_boot)
        results["positive"][str(s)] = entry

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "11_null_control.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- report
    nul = results["null"]
    print("\n================ NULL (clean vs clean, disjoint worms) ================")
    p = nul["primary_split"]
    lo, hi = p["hedges_g_ci95"]
    print(f"worm-level  g = {p['hedges_g']:+.3f} [{lo:+.2f}, {hi:+.2f}]  perm p = {p['perm_p']:.3f}  (n = {p['n_a']} vs {p['n_b']})")
    print(f"timestep KS = {nul['naive_timestep_ks']['ks']:.3f}  p = {nul['naive_timestep_ks']['p']:.2e}   <- pseudoreplicated")
    r = nul["resplits"]
    print(f"false-positive rate over {r['n_splits']} re-splits: {r['frac_p_lt_0.05']:.3f} (nominal 0.05); "
          f"95th pct |g| under null: {r['abs_g_p95']:.3f}")
    print("\n================ POSITIVE (amplitude relaxation slowed by s) ================")
    print("  s     amp AR1   amp var   power(unpaired)   g_unpaired [2.5,97.5]      paired dz   paired p")
    for s in severities:
        e = results["positive"][str(s)]
        mc = e["manipulation_check"]
        if s == 1.0:
            print(f"  {s:<5} {mc['amp_ar1_median']:.4f}   {mc['amp_var_median']:.4f}   (sham / clean reference)")
            continue
        u, pr = e["unpaired_resplits"], e["paired_all_worms"]
        print(f"  {s:<5} {mc['amp_ar1_median']:.4f}   {mc['amp_var_median']:.4f}   {u['frac_p_lt_0.05']:>9.3f}        "
              f"{u['g_mean']:+.3f} [{u['g_pct']['p2.5']:+.2f},{u['g_pct']['p97.5']:+.2f}]   "
              f"{pr['cohens_dz']:+8.3f}   {pr['signflip_p']:.4f}")
    print("\nRead-out:")
    print("  - Null FPR should sit near 0.05. If the timestep KS p is tiny on clean-vs-clean data,")
    print("    timestep-level p-values carry no evidential weight in this pipeline.")
    print("  - Pre-registered direction: flattening => NEGATIVE g. Positive or ~0 g with a clear")
    print("    manipulation check means curvature at visited states is not tracking lost resilience.")
    print("  - Minimum detectable slowdown = smallest s with unpaired power >= 0.8.")
    print(f"\nWrote {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        groups = [("A clean", clean[ia]), ("B clean", clean[ib])] + [
            (f"B s={s:g}", S[s][ib]) for s in severities if s != 1.0
        ]
        for k, (name, v) in enumerate(groups):
            jit = np.random.default_rng(k).uniform(-0.15, 0.15, v.size)
            ax[0].scatter(np.full(v.size, k) + jit, v, s=10, alpha=0.6)
            ax[0].errorbar(k, v.mean(), yerr=1.96 * v.std(ddof=1) / np.sqrt(v.size), fmt="k_", capsize=6)
        ax[0].set_xticks(range(len(groups)), [g[0] for g in groups], rotation=30)
        ax[0].set_ylabel(f"per-worm {args.summary} Hessian trace")
        ax[0].set_title("Worm-level summaries (primary split)")
        sv = [s for s in severities if s != 1.0]
        ax[1].plot(sv, [results["positive"][str(s)]["unpaired_resplits"]["frac_p_lt_0.05"] for s in sv], "o-", label="power (unpaired)")
        ax[1].axhline(r["frac_p_lt_0.05"], color="r", ls="--", label="null false-positive rate")
        ax[1].axhline(0.8, color="gray", ls=":", label="0.8")
        ax[1].set_xlabel("injected slowdown s (known)")
        ax[1].set_ylabel("fraction of splits with p < 0.05")
        ax[1].set_ylim(0, 1.02)
        ax[1].legend()
        ax[1].set_title("Detection vs. known degradation")
        plt.tight_layout()
        out_png = os.path.join(args.out_dir, "11_null_control.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Wrote {out_png}")
    except ImportError:
        logger.info("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
