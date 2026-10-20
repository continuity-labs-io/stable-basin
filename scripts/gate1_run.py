"""Gate 1 run (prompt 256): cross-day repeatability of per-electrode post-stimulus response times.

Implements docs/ai/design/3_recovery_profiling_gate1.md on FS369 2025-02-20 vs 2025-02-21.
Writes output/gate1/cells.csv (one row per active electrode, matched block, day) and
output/gate1/gate1.json. No config file, Makefile target or unit tests: deliberate debt per
prompt 256, paid only if Gate 1 passes.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H5 = Path("data/ephys/finalspark/fs369data/fs369_package.hdf5")
STIM_KEY, EVENT_KEY = "fs369_wholelife_stimulations", "fs369_wholelife_events"
DAYS = ["2025-02-20", "2025-02-21"]
TUPLE = ["a1", "a2", "d1", "d2", "nb_stim_pulse", "pulse_train_period", "stim_polarity", "stim_shape"]
SWEEP_MIN_ELEC, EXPECTED_BLOCKS = 32, 32
BASELINE_S, ACTIVE_HZ = 1800.0, 0.1
POST_LO_S, POST_HI_S, N_BINS, RESIDUAL_S = 0.003, 10.0, 40, 1.0
RETURN_X_BASELINE, HALF_PEAK_FRAC = 1.5, 0.5
MAX_CENSOR_FRAC, CEILING_MIN, GATE_R = 0.30, 0.7, 0.7
SEED, N_PERM, NULL_PCTL = 0, 1000, 95
MEASURES = ["t_return", "t_half_peak"]
OUT = Path("output/gate1")
EDGES = np.geomspace(POST_LO_S, POST_HI_S, N_BINS + 1)
CENTERS, WIDTHS = np.sqrt(EDGES[:-1] * EDGES[1:]), np.diff(EDGES)


def sweep_blocks(day, rows):
    """Split one day's stimulation rows into blocks and keep blocks whose every stimulus hits >= 32 electrodes.

    Args:
        day: date string, used in messages.
        rows: stimulation rows of that day.

    Returns:
        (kept, times, electrodes): kept blocks with TUPLE columns, {block: stimulus times}, and the
        sorted electrodes stimulated in the kept blocks.
    """
    g = rows.groupby("time_of_stim")
    if (g[TUPLE].nunique() > 1).to_numpy().any():
        sys.exit(f"STOP: {day}: a time_of_stim carries more than one parameter tuple")
    ev = g.agg(**{c: (c, "first") for c in TUPLE}, n_elec=("electrode", "nunique")).reset_index()
    ev["block"] = (ev[TUPLE] != ev[TUPLE].shift()).any(axis=1).cumsum() - 1
    b = ev.groupby("block").agg(**{c: (c, "first") for c in TUPLE}, n_elec_min=("n_elec", "min")).reset_index()
    kept = b[b["n_elec_min"] >= SWEEP_MIN_ELEC]
    if kept[TUPLE].duplicated().any():
        sys.exit(f"STOP: {day}: a parameter tuple repeats across sweep blocks; matching by tuple is ambiguous")
    times = {blk: ev.loc[ev["block"] == blk, "time_of_stim"].to_numpy() for blk in kept["block"]}
    sweep_t = np.concatenate(list(times.values()))
    elec = np.sort(rows.loc[rows["time_of_stim"].isin(sweep_t), "electrode"].unique())
    print(f"{day}: blocks={len(b)} excluded (<{SWEEP_MIN_ELEC} electrodes per stimulus)={len(b) - len(kept)} "
          f"sweep blocks={len(kept)} sweep electrodes={len(elec)}")
    return kept, times, elec


def latencies(ev, electrodes, stim_times):
    """Collect per-trial event latencies in [POST_LO_S, POST_HI_S) after each stimulus.

    Args:
        ev: events with electrode and time_of_event columns.
        electrodes: electrodes to collect.
        stim_times: list over blocks of datetime64 stimulus-time arrays.

    Returns:
        dict (electrode, block index) -> list (one per trial) of latency arrays in seconds.
    """
    out, lo_ns, hi_ns = {}, int(round(POST_LO_S * 1e9)), int(round(POST_HI_S * 1e9))
    for e in electrodes:
        te = np.sort(ev.loc[ev["electrode"] == e, "time_of_event"].to_numpy().astype("datetime64[ns]").astype(np.int64))
        for k, ts in enumerate(stim_times):
            ts = ts.astype("datetime64[ns]").astype(np.int64)
            lo, hi = np.searchsorted(te, ts + lo_ns), np.searchsorted(te, ts + hi_ns)
            out[e, k] = [(te[a:z] - t) / 1e9 for a, z, t in zip(lo, hi, ts)]
    return out


def measures(trials, baseline):
    """Compute the four response measures from pooled trials on log-spaced bins.

    Args:
        trials: list of per-trial latency arrays (s).
        baseline: the electrode's baseline rate (Hz) for that day.

    Returns:
        dict with t_return, t_half_peak (+ censoring flags), peak_rate, t_peak, residual, n_trials, n_events.
    """
    n, lat = len(trials), np.concatenate(trials)
    rate = np.histogram(lat, bins=EDGES)[0] / (n * WIDTHS)
    above = np.flatnonzero(rate > RETURN_X_BASELINE * baseline)
    ret_cens = bool(above.size) and above[-1] == N_BINS - 1
    t_ret = 0.0 if not above.size else POST_HI_S if ret_cens else float(CENTERS[above[-1]])
    ip = int(np.argmax(rate))
    below = np.flatnonzero(rate[ip + 1:] <= HALF_PEAK_FRAC * rate[ip])
    t_half = float(CENTERS[ip + 1 + below[0]] - CENTERS[ip]) if below.size else POST_HI_S
    resid = np.sum(lat >= POST_HI_S - RESIDUAL_S) / (n * RESIDUAL_S) / baseline
    return dict(t_return=t_ret, t_return_censored=ret_cens, t_half_peak=t_half, t_half_peak_censored=not below.size,
                peak_rate=float(rate[ip]), t_peak=float(CENTERS[ip]), residual=float(resid), n_trials=n, n_events=lat.size)


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
    """
    return pearson(a - a.mean(axis=0), b - b.mean(axis=0))


def analyse(mat, half, sym_a1, sym, rng):
    """Run the ceiling, three correlations, shuffle null and positive control for one measure.

    Args:
        mat: {day: (electrodes, blocks) array} on all trials.
        half: {(day, "odd"|"even"): (electrodes, blocks) array}.
        sym_a1: a1 values of the symmetric (a1 == a2) blocks.
        sym: boolean block mask for a1 == a2.
        rng: numpy Generator.

    Returns:
        dict of results.
    """
    def spearman_brown(r):  # half-length -> full-length reliability
        return 2 * r / (1 + r)

    a, b = (mat[d] for d in DAYS)
    sb = {d: spearman_brown(pearson(half[d, "odd"], half[d, "even"])) for d in DAYS}
    ceil = float(np.sqrt(sb[DAYS[0]] * sb[DAYS[1]])) if min(sb.values()) > 0 else min(sb.values())
    sb_c = {d: spearman_brown(centered_r(half[d, "odd"], half[d, "even"])) for d in DAYS}
    per_block = [pearson(a[:, k], b[:, k]) for k in range(a.shape[1])]
    slopes = {d: np.polyfit(sym_a1, mat[d][:, sym].T, 1)[0] for d in DAYS}
    obs = centered_r(a, b)
    null = np.array([centered_r(a, b[rng.permutation(a.shape[0])]) for _ in range(N_PERM)])
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
                null_p95=float(np.nanpercentile(null, NULL_PCTL)), null_frac_ge_observed=float(np.mean(null >= obs)),
                positive_control=pc)


def main():
    with pd.HDFStore(H5, mode="r") as store:
        stim, sweeps = store.select(STIM_KEY), {}
        for day in DAYS:
            rows = stim[stim["time_of_stim"].dt.normalize() == pd.Timestamp(day)]
            if rows.empty or rows[["electrode", "time_of_stim", *TUPLE]].isna().to_numpy().any():
                sys.exit(f"STOP: {day}: no stimulation rows, or NaN in electrode / time_of_stim / parameter tuple")
            sweeps[day] = sweep_blocks(day, rows)
        pairs = sweeps[DAYS[0]][0].merge(sweeps[DAYS[1]][0], on=TUPLE, suffixes=("_1", "_2"))
        if len(pairs) != EXPECTED_BLOCKS:
            sys.exit(f"STOP: matched blocks = {len(pairs)}, expected {EXPECTED_BLOCKS}")
        electrodes = np.intersect1d(sweeps[DAYS[0]][2], sweeps[DAYS[1]][2])
        base, lat, n_trials, windows = {}, {}, {}, {}
        for i, day in enumerate(DAYS):
            times = [sweeps[day][1][blk] for blk in pairs[f"block_{i + 1}"]]
            n_trials[day] = [len(t) for t in times]
            t0, t1 = pd.Timestamp(times[0].min()), pd.Timestamp(np.concatenate(times).max())
            b0, t_end = t0 - pd.to_timedelta(BASELINE_S, unit="s"), t1 + pd.to_timedelta(POST_HI_S, unit="s")
            if stim["time_of_stim"].between(b0, t0, inclusive="left").any():
                sys.exit(f"STOP: {day}: stimulation inside the baseline window {b0} - {t0}")
            cols = ["electrode", "time_of_event"]
            bl = store.select(EVENT_KEY, where=f"time_of_event >= '{b0}' & time_of_event < '{t0}'", columns=cols)
            sw = store.select(EVENT_KEY, where=f"time_of_event >= '{t0}' & time_of_event <= '{t_end}'", columns=cols)
            base[day] = bl["electrode"].value_counts().reindex(electrodes, fill_value=0) / BASELINE_S
            windows[day] = dict(baseline=[str(b0), str(t0)], sweep=[str(t0), str(t_end)],
                                baseline_events=len(bl), sweep_events=len(sw))
            print(f"{day}: baseline {b0} - {t0} events={len(bl)}; sweep {t0} - {t_end} events={len(sw)} "
                  f"(off sweep electrodes {int((~sw['electrode'].isin(electrodes)).sum())}); trials per block "
                  f"{sorted(set(n_trials[day]))}")
            lat[day] = (sw, times)
    active = electrodes[(base[DAYS[0]] > ACTIVE_HZ).to_numpy() & (base[DAYS[1]] > ACTIVE_HZ).to_numpy()]
    if active.size < 3:
        sys.exit(f"STOP: only {active.size} active electrodes")
    rows, half_rows = [], []
    for day in DAYS:
        cells = latencies(lat[day][0], active, lat[day][1])
        for (e, k), trials in cells.items():
            bl_hz, key = float(base[day][e]), dict(day=day, electrode=int(e), block=k)
            rows.append({**key, **pairs.loc[k, TUPLE].to_dict(), "baseline_hz": bl_hz, **measures(trials, bl_hz)})
            for split, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
                half_rows.append({**key, "split": split, **measures(trials[sl], bl_hz)})
    cells, halves = pd.DataFrame(rows), pd.DataFrame(half_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "cells.csv", index=False)

    def grid(df, m):
        return df.pivot(index="electrode", columns="block", values=m).loc[active].to_numpy()

    cens = {m: float(cells[f"{m}_censored"].mean()) for m in MEASURES}
    over = [m for m in MEASURES if cens[m] > MAX_CENSOR_FRAC]
    pct = ", ".join(f"{m} {100 * cens[m]:.1f}%" for m in MEASURES)
    if not over:
        primary, reason = "t_return", f"design-doc primary; censoring {pct}, both <= {MAX_CENSOR_FRAC:.0%}"
    elif len(over) == 1:
        primary = next(m for m in MEASURES if m not in over)
        reason = f"{over[0]} censored in > {MAX_CENSOR_FRAC:.0%} of cells, no usable variance; censoring {pct}"
    else:
        primary = min(MEASURES, key=cens.get)
        reason = f"both censored > {MAX_CENSOR_FRAC:.0%}, lower-censored used; censoring {pct}"
    sym = (pairs["a1"] == pairs["a2"]).to_numpy()
    rng, res = np.random.default_rng(SEED), {}
    for m in [primary, *[x for x in MEASURES if x != primary]]:
        mat = {d: grid(cells[cells["day"] == d], m) for d in DAYS}
        half = {(d, s): grid(halves[(halves["day"] == d) & (halves["split"] == s)], m) for d in DAYS for s in ("odd", "even")}
        res[m] = analyse(mat, half, pairs.loc[sym, "a1"].to_numpy(), sym, rng)
    p = res[primary]
    c2, ceil, pc = p["condition_centered_r"], p["ceiling"], p["positive_control"]["r"]
    verdict = "PASS" if c2 >= GATE_R else "UNINTERPRETABLE" if (np.isnan(c2) or ceil < CEILING_MIN) else "FAIL"
    trials_txt = "/".join(str(t) for t in sorted({t for d in DAYS for t in n_trials[d]}))
    n_line = f"n = 1 culture, {active.size} electrodes, {len(pairs)} blocks, 2 sessions, {trials_txt} trials"
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    json.dump(dict(
        code_version=rev + ("-dirty" if dirty else ""), prompt="docs/ai/complete/256_run_gate1.md", h5=str(H5),
        days=DAYS, seed=SEED,
        params=dict(tuple=TUPLE, sweep_min_electrodes=SWEEP_MIN_ELEC, baseline_s=BASELINE_S, active_hz=ACTIVE_HZ,
                    post_lo_s=POST_LO_S, post_hi_s=POST_HI_S, n_bins=N_BINS, residual_s=RESIDUAL_S,
                    return_x_baseline=RETURN_X_BASELINE, half_peak_frac=HALF_PEAK_FRAC,
                    max_censor_frac=MAX_CENSOR_FRAC, ceiling_min=CEILING_MIN, gate_r=GATE_R, n_perm=N_PERM,
                    null_percentile=NULL_PCTL, split="odd = trials 1,3,5,7,9; even = trials 2,4,6,8,10",
                    ceiling_definition="sqrt(day1 * day2 Spearman-Brown split-half), raw cells; min() if either <= 0",
                    positive_control="synthetic day 2 halves = day 1 + N(0, sd), sd set so corrected split-half"
                                     " equals the ceiling; correlation 2 of day 1 vs mean of halves"),
        windows=windows, n_trials=n_trials, matched_blocks=len(pairs), blocks=pairs.to_dict(orient="records"),
        baseline_hz={d: {int(e): float(v) for e, v in base[d].items()} for d in DAYS},
        active_electrodes=[int(e) for e in active], censoring_rate=cens, primary_measure=primary,
        primary_reason=reason, zero_event_cells=int((cells["n_events"] == 0).sum()), results=res, gate1=verdict,
        independence=n_line), open(OUT / "gate1.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'cells.csv'} ({len(cells)} rows) and {OUT / 'gate1.json'}")
    print(f"censoring: {pct}" + "".join(f"\n  {m}: NO USABLE VARIANCE (> {MAX_CENSOR_FRAC:.0%} censored)" for m in over))
    for m, r in res.items():
        sbd, sbc = r["ceiling_by_day"], r["ceiling_condition_centered_by_day"]
        print(f"[{m}{' PRIMARY' if m == primary else ''}] ceiling {r['ceiling']:.3f} (day1 {sbd[DAYS[0]]:.3f}, "
              f"day2 {sbd[DAYS[1]]:.3f}); condition-centered split-half day1 {sbc[DAYS[0]]:.3f} day2 {sbc[DAYS[1]]:.3f}")
        print(f"  per-block r median {r['per_block_median']:.3f} range {r['per_block_min']:.3f}-{r['per_block_max']:.3f}"
              f" (NaN {r['per_block_n_nan']}); condition-centered r {r['condition_centered_r']:.3f}; raw pooled r "
              f"{r['raw_pooled_r_not_headline']:.3f} (not headline); dose-slope r {r['dose_slope_r']:.3f}")
        print(f"  null p95 {r['null_p95']:.3f}, fraction of null >= observed {r['null_frac_ge_observed']:.3f}; "
              f"positive control r {r['positive_control']['r']:.3f} (synthetic split-half "
              f"{r['positive_control']['synthetic_split_half']:.3f} vs ceiling {r['ceiling']:.3f}; expected raw r "
              f"sqrt(ceiling) {r['positive_control']['expected_raw_r']:.3f} since day 1 enters noise-free)")
    if ceil < CEILING_MIN:
        print("CEILING TOO LOW")
    print(n_line)
    print(f"PRIMARY MEASURE: {primary} ({reason})\nACTIVE ELECTRODES: {active.size}\nMATCHED BLOCKS: {len(pairs)}")
    print(f"CEILING: {ceil:.3f} (day1 {p['ceiling_by_day'][DAYS[0]]:.3f}, day2 {p['ceiling_by_day'][DAYS[1]]:.3f})")
    print(f"PER BLOCK r: median {p['per_block_median']:.3f} range {p['per_block_min']:.3f}-{p['per_block_max']:.3f}")
    print(f"CONDITION-CENTERED r: {c2:.3f}  null p95 {p['null_p95']:.3f}  positive control {pc:.3f}")
    print(f"GATE 1: {verdict}")


if __name__ == "__main__":
    main()
