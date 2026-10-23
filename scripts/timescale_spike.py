"""THROWAWAY reconnaissance (prompt 258): within-session split-half reliability per timescale band.

Recomputes per-trial latencies for the Gate 1 cells (FS369 2025-02-20 / 2025-02-21, matched sweep
blocks, active electrodes), splits each cell's 10 trials into odd/even halves, computes eight response
measures per half, correlates halves across cells after per-block centering, and applies Spearman-Brown.
Prints to stdout only. No tests, config or output files, per prompt 258. Run from the repository root.
"""

import sys

import numpy as np
import pandas as pd

from gate1_run import (ACTIVE_HZ, BASELINE_S, DAYS, EDGES, EVENT_KEY, EXPECTED_BLOCKS, H5, POST_HI_S, STIM_KEY,
                       TUPLE, WIDTHS, centered_r, latencies, measures, pearson, sweep_blocks)

WINDOWS = dict(early_amp=(0.003, 0.060), hump_amp=(0.3, 2.0), late_amp=(2.0, 10.0))  # seconds, from prompt 258
BANDS = dict(t_return="slow", t_half_peak="fast", peak_rate="fast", residual="slow",
             t_peak="fast", early_amp="fast", hump_amp="mid", late_amp="slow")  # hardcoded labels, prompt 258
GATE1_CEILING = dict(t_return=0.779, t_half_peak=0.156)  # output/gate1/stdout.txt, raw sqrt(d1 * d2) ceiling
SANITY_TOL, UNUSABLE_FRAC = 0.05, 0.30
SPLITS = dict(odd=slice(0, None, 2), even=slice(1, None, 2))  # odd = trials 1,3,5,7,9 as in gate1_run.py


def spearman_brown(r):
    """Step a half-length correlation up to full-length reliability.

    Args:
        r: split-half correlation.

    Returns:
        2r / (1 + r).
    """
    return 2 * r / (1 + r)


def all_measures(trials, baseline):
    """Compute the eight measures for one cell: five imported from gate1_run.measures, three window amplitudes.

    Args:
        trials: list of per-trial latency arrays (s).
        baseline: the electrode's baseline rate (Hz) on that day.

    Returns:
        dict measure -> value, plus n_events and the gate1 censoring flags.
    """
    m, n, lat = measures(trials, baseline), len(trials), np.concatenate(trials)
    for name, (lo, hi) in WINDOWS.items():  # event-count mean rate in [lo, hi), minus baseline
        m[name] = float(np.sum((lat >= lo) & (lat < hi)) / (n * (hi - lo)) - baseline)
    return m


def undefined(df, name):
    """Flag cells where a measure has no defined value.

    Args:
        df: full-trial cell table.
        name: measure name.

    Returns:
        boolean Series: gate1 censoring flag if one exists, no events for t_peak, else NaN.
    """
    if f"{name}_censored" in df:
        return df[f"{name}_censored"].astype(bool)
    if name == "t_peak":
        return df["n_events"] == 0
    return df[name].isna()


def load():
    """Load the Gate 1 matched blocks, baselines, active electrodes and sweep events (copied from gate1_run.main).

    Returns:
        (active, base, lat): active electrode ids, {day: baseline Hz Series}, {day: (sweep events, block times)}.
    """
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
        base, lat = {}, {}
        for i, day in enumerate(DAYS):
            times = [sweeps[day][1][blk] for blk in pairs[f"block_{i + 1}"]]
            t0, t1 = pd.Timestamp(times[0].min()), pd.Timestamp(np.concatenate(times).max())
            b0, t_end = t0 - pd.to_timedelta(BASELINE_S, unit="s"), t1 + pd.to_timedelta(POST_HI_S, unit="s")
            if stim["time_of_stim"].between(b0, t0, inclusive="left").any():
                sys.exit(f"STOP: {day}: stimulation inside the baseline window {b0} - {t0}")
            cols = ["electrode", "time_of_event"]
            bl = store.select(EVENT_KEY, where=f"time_of_event >= '{b0}' & time_of_event < '{t0}'", columns=cols)
            sw = store.select(EVENT_KEY, where=f"time_of_event >= '{t0}' & time_of_event <= '{t_end}'", columns=cols)
            base[day] = bl["electrode"].value_counts().reindex(electrodes, fill_value=0) / BASELINE_S
            lat[day] = (sw, times)
    active = electrodes[(base[DAYS[0]] > ACTIVE_HZ).to_numpy() & (base[DAYS[1]] > ACTIVE_HZ).to_numpy()]
    if active.size < 3:
        sys.exit(f"STOP: only {active.size} active electrodes")
    return active, base, lat


def main():
    print("NOTE: data loading copied from gate1_run.main (not importable as a function); measures, latencies, "
          "sweep_blocks, centered_r imported unchanged")
    print("NOTE: residual is gate1_run's definition (event rate in last 1.0 s / baseline), not final-bin minus baseline")
    active, base, lat = load()
    full, half = [], []
    for day in DAYS:
        for (e, k), trials in latencies(lat[day][0], active, lat[day][1]).items():
            key, bl_hz = dict(day=day, electrode=int(e), block=k), float(base[day][e])
            full.append({**key, **all_measures(trials, bl_hz)})
            half += [{**key, "split": s, **all_measures(trials[sl], bl_hz)} for s, sl in SPLITS.items()]
    full, half = pd.DataFrame(full), pd.DataFrame(half)

    def grid(day, split, name):
        df = half[(half["day"] == day) & (half["split"] == split)]
        return df.pivot(index="electrode", columns="block", values=name).loc[active].to_numpy()

    ceilings, raw = {}, {}
    for name, band in BANDS.items():
        ceilings[name] = [spearman_brown(centered_r(grid(d, "odd", name), grid(d, "even", name))) for d in DAYS]
        raw[name] = [spearman_brown(pearson(grid(d, "odd", name), grid(d, "even", name))) for d in DAYS]
        frac = float(undefined(full, name).mean())
        flag = " UNUSABLE" if frac > UNUSABLE_FRAC else ""
        c1, c2 = ceilings[name]
        print(f"MEASURE {name:<13} band {band:<4}  ceiling_d1 {c1:.3f}  ceiling_d2 {c2:.3f}  "
              f"undefined_frac {frac:.2f}{flag}")
        if flag:
            ceilings[name] = None
    n_blocks = full["block"].nunique()
    print(f"CELLS: {active.size} electrodes x {n_blocks} blocks x {len(DAYS)} days")
    medians = {b: float(np.nanmedian([c for n, bb in BANDS.items() if bb == b and ceilings[n] for c in ceilings[n]]))
               for b in ("fast", "mid", "slow")}
    for name, ref in GATE1_CEILING.items():  # gate1 "ceiling" = sqrt(d1 * d2) of raw (uncentered) split-half
        r1, r2 = raw[name]
        got = float(np.sqrt(r1 * r2)) if min(r1, r2) > 0 else min(r1, r2)
        ok = "ok" if abs(got - ref) <= SANITY_TOL else f"MISMATCH > {SANITY_TOL}: reimplementation suspect"
        print(f"SANITY {name}: raw split-half ceiling {got:.3f} (d1 {r1:.3f}, d2 {r2:.3f}) vs Gate 1 {ref:.3f} {ok}")
    print("BAND MEDIAN CEILING: " + "  ".join(f"{b} {v:.2f}" for b, v in medians.items()))
    print(f"VERDICT: {max(medians, key=lambda b: -np.inf if np.isnan(medians[b]) else medians[b])} "
          "carries the reproducible signal")


if __name__ == "__main__":
    main()
