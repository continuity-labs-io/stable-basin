import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def compute_stats(y, o):
    """
    Computes statistical significance metrics between two cohorts.
    Returns a dictionary of statistics including Welch's t-test,
    Mann-Whitney U test, and Cohen's d effect size.
    """
    t_stat, t_p = ttest_ind(y, o, equal_var=False)
    u_stat, u_p = mannwhitneyu(y, o)
    nx = len(y)
    ny = len(o)
    dof = nx + ny - 2
    if dof <= 0:
        return {"welch_t_stat": 0.0, "welch_p_value": 1.0, "mann_whitney_u_stat": 0.0, "mann_whitney_p_value": 1.0, "cohens_d": 0.0}
    pool_sd = np.sqrt(((nx - 1) * np.var(y, ddof=1) + (ny - 1) * np.var(o, ddof=1)) / dof)
    d = (np.mean(y) - np.mean(o)) / pool_sd if pool_sd > 0 else 0.0
    return {
        "welch_t_stat": float(t_stat),
        "welch_p_value": float(t_p),
        "mann_whitney_u_stat": float(u_stat),
        "mann_whitney_p_value": float(u_p),
        "cohens_d": float(d)
    }

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
    from scipy.stats import ks_2samp
    ks = ks_2samp(a, b)
    return {"n_a": int(a.size), "n_b": int(b.size), "ks": float(ks.statistic), "p": float(ks.pvalue),
            "note": "pseudoreplicated: timesteps treated as independent; shown for comparison only"}

def resplit_stats(clean_s, cond_s, labels, n_splits, seed) -> dict:
    from src.data.utils import stratified_split
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

