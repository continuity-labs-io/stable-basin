"""Gate 1 analysis functions: per-electrode post-stimulus response measures and cross-day correlations.

Moved out of scripts/gate1_run.py so the logic can be imported and unit tested. The script keeps the
HDF5 queries, the parameter values and the output writing; every function here receives its
parameters as arguments. Implements docs/ai/design/3_recovery_profiling_gate1.md (prompt 256).
"""

import numpy as np


def sweep_blocks(rows, tuple_cols, min_electrodes):
    """Split one day's stimulation rows into blocks and keep blocks whose every stimulus hits enough electrodes.

    A block is a run of consecutive unique ``time_of_stim`` values sharing the same parameter tuple.

    Args:
        rows: stimulation rows of one day, with ``time_of_stim``, ``electrode`` and ``tuple_cols`` columns.
        tuple_cols: list of parameter column names that define a block.
        min_electrodes: minimum number of electrodes every stimulus in a block must hit to be kept.

    Returns:
        (kept, times, electrodes, n_blocks): kept blocks with ``block`` and ``tuple_cols`` columns,
        {block: stimulus times}, the sorted electrodes stimulated in the kept blocks, and the total
        number of blocks before exclusion.

    Raises:
        ValueError: if a ``time_of_stim`` carries more than one parameter tuple, or a parameter tuple
            repeats across kept blocks (so matching blocks by tuple would be ambiguous).
    """
    g = rows.groupby("time_of_stim")
    if (g[tuple_cols].nunique() > 1).to_numpy().any():
        raise ValueError("a time_of_stim carries more than one parameter tuple")
    ev = g.agg(**{c: (c, "first") for c in tuple_cols}, n_elec=("electrode", "nunique")).reset_index()
    ev["block"] = (ev[tuple_cols] != ev[tuple_cols].shift()).any(axis=1).cumsum() - 1
    b = ev.groupby("block").agg(**{c: (c, "first") for c in tuple_cols}, n_elec_min=("n_elec", "min")).reset_index()
    kept = b[b["n_elec_min"] >= min_electrodes]
    if kept[tuple_cols].duplicated().any():
        raise ValueError("a parameter tuple repeats across sweep blocks; matching by tuple is ambiguous")
    times = {blk: ev.loc[ev["block"] == blk, "time_of_stim"].to_numpy() for blk in kept["block"]}
    sweep_t = np.concatenate(list(times.values())) if times else np.array([], dtype="datetime64[ns]")
    elec = np.sort(rows.loc[rows["time_of_stim"].isin(sweep_t), "electrode"].unique())
    return kept, times, elec, len(b)


def latencies(ev, electrodes, stim_times, post_lo_s, post_hi_s):
    """Collect per-trial event latencies in [post_lo_s, post_hi_s) after each stimulus.

    Args:
        ev: events with ``electrode`` and ``time_of_event`` columns.
        electrodes: electrodes to collect.
        stim_times: list over blocks of datetime64 stimulus-time arrays.
        post_lo_s: start of the post-stimulus window (s), inclusive.
        post_hi_s: end of the post-stimulus window (s), exclusive.

    Returns:
        dict (electrode, block index) -> list (one per trial) of latency arrays in seconds.
    """
    out, lo_ns, hi_ns = {}, int(round(post_lo_s * 1e9)), int(round(post_hi_s * 1e9))
    for e in electrodes:
        te = np.sort(ev.loc[ev["electrode"] == e, "time_of_event"].to_numpy().astype("datetime64[ns]").astype(np.int64))
        for k, ts in enumerate(stim_times):
            ts = ts.astype("datetime64[ns]").astype(np.int64)
            lo, hi = np.searchsorted(te, ts + lo_ns), np.searchsorted(te, ts + hi_ns)
            out[e, k] = [(te[a:z] - t) / 1e9 for a, z, t in zip(lo, hi, ts)]
    return out


def measures(trials, baseline, edges, return_x_baseline, half_peak_frac, residual_s):
    """Compute the response measures from pooled trials on the given (log-spaced) bins.

    ``t_return`` is the centre of the last bin whose rate exceeds ``return_x_baseline * baseline``
    (0 if none; ``edges[-1]`` and censored if the final bin exceeds it). ``t_half_peak`` is the time
    from the peak bin to the first later bin at or below ``half_peak_frac * peak_rate`` (``edges[-1]``
    and censored if never reached). ``residual`` is the rate over the final ``residual_s`` seconds of
    the window as a multiple of baseline.

    Args:
        trials: list of per-trial latency arrays (s).
        baseline: the electrode's baseline rate (Hz) for that day.
        edges: increasing bin edges (s); bin centres are geometric means of adjacent edges.
        return_x_baseline: multiple of baseline that defines ``t_return``.
        half_peak_frac: fraction of the peak rate that defines ``t_half_peak``.
        residual_s: length (s) of the final window used for ``residual``.

    Returns:
        dict with t_return, t_return_censored, t_half_peak, t_half_peak_censored, peak_rate, t_peak,
        residual, n_trials, n_events.
    """
    edges = np.asarray(edges, dtype=float)
    centers, widths, post_hi_s, n_bins = np.sqrt(edges[:-1] * edges[1:]), np.diff(edges), edges[-1], edges.size - 1
    n, lat = len(trials), np.concatenate(trials)
    rate = np.histogram(lat, bins=edges)[0] / (n * widths)
    above = np.flatnonzero(rate > return_x_baseline * baseline)
    ret_cens = bool(above.size) and above[-1] == n_bins - 1
    t_ret = 0.0 if not above.size else float(post_hi_s) if ret_cens else float(centers[above[-1]])
    ip = int(np.argmax(rate))
    below = np.flatnonzero(rate[ip + 1:] <= half_peak_frac * rate[ip])
    t_half = float(centers[ip + 1 + below[0]] - centers[ip]) if below.size else float(post_hi_s)
    resid = np.sum(lat >= post_hi_s - residual_s) / (n * residual_s) / baseline
    return dict(t_return=t_ret, t_return_censored=ret_cens, t_half_peak=t_half, t_half_peak_censored=not below.size,
                peak_rate=float(rate[ip]), t_peak=float(centers[ip]), residual=float(resid), n_trials=n,
                n_events=lat.size)


def pearson(x, y):
    """Pearson r of two flattened arrays; NaN if either is constant.

    Args:
        x: array.
        y: array of the same size.

    Returns:
        float correlation.
    """
    x, y = np.ravel(x), np.ravel(y)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def centered_r(a, b):
    """Correlation 2: subtract each block's mean over electrodes within each day, then correlate.

    Args:
        a: (electrodes, blocks) day-1 values.
        b: (electrodes, blocks) day-2 values.

    Returns:
        float correlation of the residuals.

    Raises:
        ValueError: if ``a`` and ``b`` differ in shape.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return pearson(a - a.mean(axis=0), b - b.mean(axis=0))


def spearman_brown(r):
    """Spearman-Brown correction from half-length to full-length reliability.

    Args:
        r: split-half correlation.

    Returns:
        float corrected reliability ``2r / (1 + r)``.
    """
    return 2 * r / (1 + r)


def analyse(mat, half, days, sym_a1, sym, rng, n_perm, null_pctl):
    """Run the ceiling, three correlations, shuffle null and positive control for one measure.

    Args:
        mat: {day: (electrodes, blocks) array} on all trials.
        half: {(day, "odd"|"even"): (electrodes, blocks) array}.
        days: the two day keys, in order.
        sym_a1: a1 values of the symmetric (a1 == a2) blocks.
        sym: boolean block mask for a1 == a2.
        rng: numpy Generator.
        n_perm: number of electrode-label permutations for the null.
        null_pctl: percentile of the null distribution to report.

    Returns:
        dict of results.
    """
    a, b = (mat[d] for d in days)
    sb = {d: spearman_brown(pearson(half[d, "odd"], half[d, "even"])) for d in days}
    ceil = float(np.sqrt(sb[days[0]] * sb[days[1]])) if min(sb.values()) > 0 else min(sb.values())
    sb_c = {d: spearman_brown(centered_r(half[d, "odd"], half[d, "even"])) for d in days}
    per_block = [pearson(a[:, k], b[:, k]) for k in range(a.shape[1])]
    slopes = {d: np.polyfit(sym_a1, mat[d][:, sym].T, 1)[0] for d in days}
    obs = centered_r(a, b)
    null = np.array([centered_r(a, b[rng.permutation(a.shape[0])]) for _ in range(n_perm)])
    pc = dict(r=float("nan"), noise_sd=float("nan"), synthetic_split_half=float("nan"),
              expected_raw_r=float(np.sqrt(ceil)) if ceil > 0 else float("nan"))  # day 1 enters noise-free
    if 0 < ceil < 1:  # synthetic day 2 halves = day 1 + N(0, sd); sd set so corrected split-half == ceiling
        sd = float(np.sqrt(a.var() * (1 / (ceil / (2 - ceil)) - 1)))
        odd, even = a + rng.normal(0, sd, a.shape), a + rng.normal(0, sd, a.shape)
        pc.update(r=centered_r(a, (odd + even) / 2), noise_sd=sd,
                  synthetic_split_half=spearman_brown(pearson(odd, even)))
    return dict(ceiling=ceil, ceiling_by_day=sb, ceiling_condition_centered_by_day=sb_c,
                per_block_r=per_block, per_block_median=float(np.nanmedian(per_block)),
                per_block_min=float(np.nanmin(per_block)), per_block_max=float(np.nanmax(per_block)),
                per_block_n_nan=int(np.isnan(per_block).sum()), condition_centered_r=obs,
                raw_pooled_r_not_headline=pearson(a, b), dose_slope_r=pearson(*slopes.values()),
                null_p95=float(np.nanpercentile(null, null_pctl)), null_frac_ge_observed=float(np.mean(null >= obs)),
                positive_control=pc)
