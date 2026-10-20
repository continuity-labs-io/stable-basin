"""Read-only stimulation-protocol report for a FinalSpark whole-life package.

Reads `<fs_id>_package.hdf5` in read-only mode and prints a text report. It
does not plot and does not write any files.

Tasks:
    1. Row semantics: group rows on the same electrode separated by gaps
       <= `--group-gap-s` and compare group size with `nb_stim_pulse`.
    2. `pulse_train_period` units: if rows are pulses, compare the observed
       within-group gap with `pulse_train_period`; otherwise print the
       trigger-to-trigger gap distribution and stop (no unit inferred).
    3. Amplitude vs. culture age: per-day summary of `a1` and Spearman
       correlation of `a1` against days since first stimulation.
    4. Parameter sets: distinct (a1, a2, d1, d2, nb_stim_pulse,
       pulse_train_period, stim_polarity, stim_shape) tuples and coverage.
    5. Unstimulated epochs: pre-stimulation baseline, pooled inter-stimulus
       gaps longer than `--long-gap-h`, and post-stimulation tail.

Usage:
    python src/data/ephys/finalspark_protocol.py
    python src/data/ephys/finalspark_protocol.py --data-dir data/ephys/finalspark/fs369data
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.ephys.check_finalspark import DEFAULT_DATA_DIR, find_package

# Grouping threshold for TASK 1. Within-electrode gaps in fs369 have no
# values in (0.06, 0.1] s, so 0.08 s separates sub-100 ms bursts from the
# >= 0.1 s trigger spacing. Override with --group-gap-s.
DEFAULT_GROUP_GAP_S = 0.08
# Fraction of groups whose size must equal nb_stim_pulse for "ROWS ARE PULSES".
DEFAULT_MATCH_FRACTION = 0.9
# |rho| at or above which a1 is reported as confounded with age (from prompt 252).
DEFAULT_RHO_THRESHOLD = 0.3
# Minimum pooled inter-stimulus gap reported as an unstimulated epoch (prompt 252).
DEFAULT_LONG_GAP_H = 6.0
DEFAULT_TOP_SETS = 15
DEFAULT_TOP_GAPS = 10
# Rows read per chunk when scanning the events table for its time range.
EVENT_CHUNK_ROWS = 10_000_000

PARAM_COLS = [
    "a1",
    "a2",
    "d1",
    "d2",
    "nb_stim_pulse",
    "pulse_train_period",
    "stim_polarity",
    "stim_shape",
]
# Observed-seconds / pulse_train_period ratio -> implied unit.
UNIT_RATIOS = {"microseconds": 1e-6, "milliseconds": 1e-3, "seconds": 1.0}
GAP_HIST_EDGES_S = [0, 0.001, 0.01, 0.02, 0.06, 0.1, 0.5, 1, 2, 5, 10, 20, 60, 3600, np.inf]


def load_tables(pkg: str, fs_id: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load the stimulation table and the events time range, read-only.

    Args:
        pkg: Path to the `<fs_id>_package.hdf5` file.
        fs_id: Culture ID prefix used in the store keys (e.g. "fs369").

    Returns:
        Tuple of (stimulation DataFrame with datetime `time_of_stim`, first
        event timestamp, last event timestamp).
    """
    ev_key = f"{fs_id}_wholelife_events"
    with pd.HDFStore(pkg, mode="r") as store:
        stim = store.select(f"{fs_id}_wholelife_stimulations")
        n_ev = store.get_storer(ev_key).nrows
        t_min, t_max = None, None
        for start in range(0, n_ev, EVENT_CHUNK_ROWS):
            col = store.select_column(
                ev_key, "time_of_event", start=start, stop=min(start + EVENT_CHUNK_ROWS, n_ev)
            )
            col = pd.to_datetime(col)
            lo, hi = col.min(), col.max()
            t_min = lo if t_min is None or lo < t_min else t_min
            t_max = hi if t_max is None or hi > t_max else t_max
    stim = stim.copy()
    stim["time_of_stim"] = pd.to_datetime(stim["time_of_stim"])
    return stim, t_min, t_max


def assign_groups(stim: pd.DataFrame, group_gap_s: float) -> pd.DataFrame:
    """Group rows on the same electrode separated by gaps <= `group_gap_s`.

    Args:
        stim: Stimulation table with `electrode` and datetime `time_of_stim`.
        group_gap_s: Maximum within-group gap between consecutive rows (s).

    Returns:
        Copy of `stim` sorted by (electrode, time_of_stim) with added columns
        `gap_s` (gap to previous row on the same electrode, NaN for the first)
        and `group_id` (int, unique across electrodes).

    Raises:
        ValueError: If `group_gap_s` is not positive.
    """
    if group_gap_s <= 0:
        raise ValueError(f"group_gap_s must be positive, got {group_gap_s}")
    df = stim.sort_values(["electrode", "time_of_stim"], kind="mergesort").reset_index(drop=True)
    df["gap_s"] = df.groupby("electrode")["time_of_stim"].diff().dt.total_seconds()
    new_group = df["gap_s"].isna() | (df["gap_s"] > group_gap_s)
    df["group_id"] = new_group.cumsum() - 1
    return df


def row_semantics(
    stim: pd.DataFrame, group_gap_s: float, match_fraction: float
) -> dict:
    """TASK 1: decide whether rows are single pulses or whole triggers.

    Args:
        stim: Stimulation table.
        group_gap_s: Maximum within-group gap (s) passed to `assign_groups`.
        match_fraction: Minimum fraction of groups whose size equals the
            group's `nb_stim_pulse` for the verdict "ROWS ARE PULSES".

    Returns:
        Dict with keys "grouped" (DataFrame from `assign_groups`), "groups"
        (per-group DataFrame with `size`, `nb_stim_pulse`, `pulse_train_period`,
        `median_gap_s`), "median_size" (dict nb_stim_pulse -> median size),
        "match_frac" (float), "rows_are_pulses" (bool), "verdict" (str).
    """
    df = assign_groups(stim, group_gap_s)
    groups = df.groupby("group_id").agg(
        size=("time_of_stim", "size"),
        nb_stim_pulse=("nb_stim_pulse", lambda s: s.mode().iloc[0]),
        n_nb_values=("nb_stim_pulse", "nunique"),
        pulse_train_period=("pulse_train_period", lambda s: s.mode().iloc[0]),
        median_gap_s=("gap_s", lambda s: s.iloc[1:].median() if len(s) > 1 else np.nan),
    )
    median_size = {
        float(nb): float(g["size"].median()) for nb, g in groups.groupby("nb_stim_pulse")
    }
    match = groups["size"] == groups["nb_stim_pulse"]
    match_frac = float(match.mean()) if len(groups) else 0.0
    medians_match = all(med == nb for nb, med in median_size.items())
    rows_are_pulses = bool(medians_match and match_frac >= match_fraction)
    return {
        "grouped": df,
        "groups": groups,
        "median_size": median_size,
        "match_frac": match_frac,
        "rows_are_pulses": rows_are_pulses,
        "verdict": "ROWS ARE PULSES" if rows_are_pulses else "ROWS ARE TRIGGERS",
    }


def implied_unit(ratio: float) -> str:
    """Map an observed-seconds / pulse_train_period ratio to the nearest unit.

    Args:
        ratio: Observed seconds divided by the raw `pulse_train_period` value.

    Returns:
        Name of the unit in `UNIT_RATIOS` nearest in log10 distance, or
        "undetermined" if `ratio` is not a positive finite number.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return "undetermined"
    return min(UNIT_RATIOS, key=lambda u: abs(np.log10(ratio) - np.log10(UNIT_RATIOS[u])))


def train_period_units(task1: dict) -> dict:
    """TASK 2: resolve `pulse_train_period` units where observable.

    Args:
        task1: Output of `row_semantics`.

    Returns:
        Dict with key "observable" (bool). If True, also "ratios" (dict
        pulse_train_period -> (observed median seconds, ratio, unit)). If
        False, also "gap_hist" (DataFrame: gap bins x pulse_train_period
        counts of consecutive-trigger gaps on the same electrode).
    """
    if task1["rows_are_pulses"]:
        groups = task1["groups"].dropna(subset=["median_gap_s"])
        ratios = {}
        for period, g in groups.groupby("pulse_train_period"):
            obs = float(g["median_gap_s"].median())
            ratio = obs / float(period)
            ratios[float(period)] = (obs, ratio, implied_unit(ratio))
        return {"observable": True, "ratios": ratios}
    df = task1["grouped"].dropna(subset=["gap_s"])
    bins = pd.cut(df["gap_s"], GAP_HIST_EDGES_S, right=True)
    hist = pd.crosstab(bins, df["pulse_train_period"])
    return {"observable": False, "gap_hist": hist}


def dose_vs_age(stim: pd.DataFrame, rho_threshold: float) -> dict:
    """TASK 3: per-day a1 summary and Spearman(a1, days since first stim).

    Args:
        stim: Stimulation table.
        rho_threshold: |rho| at or above which the verdict is
            "DOSE CONFOUNDED WITH AGE".

    Returns:
        Dict with keys "daily" (DataFrame indexed by integer day since first
        stimulation), "rho" (float), "p" (float), "verdict" (str).
    """
    t0 = stim["time_of_stim"].min()
    days = (stim["time_of_stim"] - t0).dt.total_seconds() / 86400.0
    day_bin = np.floor(days).astype(int)
    df = stim.assign(day=day_bin)
    daily = df.groupby("day").agg(
        n_stims=("a1", "size"),
        n_electrodes=("electrode", "nunique"),
        median_a1=("a1", "median"),
        iqr_a1=("a1", lambda s: float(s.quantile(0.75) - s.quantile(0.25))),
        d1_values=("d1", lambda s: sorted(s.unique().tolist())),
        nb_stim_pulse_values=("nb_stim_pulse", lambda s: sorted(s.unique().tolist())),
    )
    rho, p = spearmanr(stim["a1"].to_numpy(), days.to_numpy())
    rho, p = float(rho), float(p)
    confounded = bool(np.isfinite(rho) and abs(rho) >= rho_threshold)
    return {
        "daily": daily,
        "rho": rho,
        "p": p,
        "verdict": "DOSE CONFOUNDED WITH AGE" if confounded else "DOSE AXIS USABLE",
    }


def parameter_sets(stim: pd.DataFrame) -> dict:
    """TASK 4: distinct parameter sets and their coverage.

    Args:
        stim: Stimulation table containing all `PARAM_COLS`.

    Returns:
        Dict with keys "sets" (DataFrame indexed by the parameter tuple with
        `n_stims`, `n_electrodes`, `n_days`, `first`, `last`, sorted by
        `n_stims` descending), "n_sets" (int), "widest" (index tuple of the
        set with the most distinct days, ties broken by electrodes then stims).
    """
    day = stim["time_of_stim"].dt.floor("D")
    sets = (
        stim.assign(_day=day)
        .groupby(PARAM_COLS)
        .agg(
            n_stims=("time_of_stim", "size"),
            n_electrodes=("electrode", "nunique"),
            n_days=("_day", "nunique"),
            first=("time_of_stim", "min"),
            last=("time_of_stim", "max"),
        )
        .sort_values("n_stims", ascending=False)
    )
    widest = sets.sort_values(
        ["n_days", "n_electrodes", "n_stims"], ascending=False, kind="mergesort"
    ).index[0]
    return {"sets": sets, "n_sets": len(sets), "widest": widest}


def unstimulated_epochs(
    stim: pd.DataFrame, first_event: pd.Timestamp, last_event: pd.Timestamp, long_gap_h: float
) -> dict:
    """TASK 5: pre-stim baseline, long pooled inter-stim gaps, post-stim tail.

    Args:
        stim: Stimulation table.
        first_event: Timestamp of the first detected event.
        last_event: Timestamp of the last detected event.
        long_gap_h: Minimum pooled inter-stimulus gap (hours) to report.

    Returns:
        Dict with keys "baseline_h" (first event -> first stim, hours),
        "tail_h" (last stim -> last event, hours), "long_gaps" (DataFrame with
        `start`, `end`, `duration_h` sorted descending), "total_h" (baseline +
        tail + sum of long gaps, hours; negative baseline/tail are clipped to 0).
    """
    t = stim["time_of_stim"].sort_values().reset_index(drop=True)
    gaps_h = t.diff().dt.total_seconds() / 3600.0
    mask = gaps_h > long_gap_h
    long_gaps = pd.DataFrame(
        {"start": t.shift(1)[mask], "end": t[mask], "duration_h": gaps_h[mask]}
    ).sort_values("duration_h", ascending=False, kind="mergesort").reset_index(drop=True)
    baseline_h = (t.iloc[0] - first_event).total_seconds() / 3600.0
    tail_h = (last_event - t.iloc[-1]).total_seconds() / 3600.0
    total_h = max(baseline_h, 0.0) + max(tail_h, 0.0) + float(long_gaps["duration_h"].sum())
    return {
        "baseline_h": baseline_h,
        "tail_h": tail_h,
        "long_gaps": long_gaps,
        "total_h": total_h,
    }


def _fmt_p(p: float) -> str:
    """Format a p-value, showing float64 underflow as an upper bound.

    Args:
        p: p-value.

    Returns:
        "< 1e-300" if `p` underflowed to 0.0, else `p` in 3 significant digits.
    """
    return "< 1e-300" if p == 0.0 else f"= {p:.3g}"


def _fmt_set(key: tuple) -> str:
    """Format a parameter-set tuple as `name=value` pairs.

    Args:
        key: Tuple of values ordered as `PARAM_COLS`.

    Returns:
        Comma-separated `name=value` string.
    """
    return ", ".join(f"{c}={v:g}" for c, v in zip(PARAM_COLS, key))


def run_report(
    stim: pd.DataFrame,
    first_event: pd.Timestamp,
    last_event: pd.Timestamp,
    group_gap_s: float = DEFAULT_GROUP_GAP_S,
    match_fraction: float = DEFAULT_MATCH_FRACTION,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    long_gap_h: float = DEFAULT_LONG_GAP_H,
    top_sets: int = DEFAULT_TOP_SETS,
    top_gaps: int = DEFAULT_TOP_GAPS,
) -> list[str]:
    """Run all five tasks, print the report, and return the summary lines.

    Args:
        stim: Stimulation table with datetime `time_of_stim`.
        first_event: Timestamp of the first detected event.
        last_event: Timestamp of the last detected event.
        group_gap_s: TASK 1 grouping threshold (s).
        match_fraction: TASK 1 match fraction for "ROWS ARE PULSES".
        rho_threshold: TASK 3 |rho| threshold.
        long_gap_h: TASK 5 minimum reported gap (hours).
        top_sets: Number of parameter sets listed in TASK 4.
        top_gaps: Number of long gaps listed in TASK 5.

    Returns:
        The five summary lines, one per task, each starting with its verdict.
    """
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=== TASK 1: row semantics ===")
    t1 = row_semantics(stim, group_gap_s, match_fraction)
    gaps = t1["grouped"]["gap_s"].dropna()
    hist = pd.cut(gaps, GAP_HIST_EDGES_S).value_counts().sort_index()
    print("within-electrode gap histogram (s):")
    print("  " + "  ".join(f"{iv}:{n}" for iv, n in hist.items()))
    print(f"group gap threshold: {group_gap_s} s   groups: {len(t1['groups'])}")
    for nb, med in sorted(t1["median_size"].items()):
        n_g = int((t1["groups"]["nb_stim_pulse"] == nb).sum())
        print(f"  nb_stim_pulse=={nb:g}: groups={n_g}  median group size={med:g}")
    print(f"  groups with mixed nb_stim_pulse: {int((t1['groups']['n_nb_values'] > 1).sum())}")
    print(f"  fraction of groups with size == nb_stim_pulse: {t1['match_frac']:.4f}")
    print(t1["verdict"])

    print("\n=== TASK 2: pulse_train_period units ===")
    t2 = train_period_units(t1)
    if t2["observable"]:
        units = set()
        for period, (obs, ratio, unit) in sorted(t2["ratios"].items()):
            print(f"  period={period:g}: observed median gap={obs:.6g} s  ratio={ratio:.3g}  -> {unit}")
            units.add(unit)
        t2_line = f"UNIT {'/'.join(sorted(units))}: pulse_train_period implied by observed gaps"
    else:
        print("Rows are triggers: the within-train pulse interval is not observable in this table.")
        print("Gap between consecutive triggers on the same electrode, by pulse_train_period:")
        print(t2["gap_hist"].to_string())
        t2_line = "UNOBSERVABLE: rows are triggers, pulse_train_period unit not resolved"
    print(t2_line.split(":")[0])

    print("\n=== TASK 3: a1 vs days since first stimulation ===")
    t3 = dose_vs_age(stim, rho_threshold)
    print(f"first stim: {stim['time_of_stim'].min()}")
    print(t3["daily"].to_string())
    print(f"Spearman rho(a1, days since first stim) = {t3['rho']:.4f}  p {_fmt_p(t3['p'])}")
    print(t3["verdict"])

    print("\n=== TASK 4: parameter sets ===")
    t4 = parameter_sets(stim)
    sets = t4["sets"]
    print(f"distinct parameter sets: {t4['n_sets']}   columns: {PARAM_COLS}")
    print(f"top {top_sets} by count:")
    print(sets.head(top_sets).reset_index().to_string(index=False))
    print("distinct days per set (all sets, by count):")
    print("  " + " ".join(str(int(d)) for d in sets["n_days"]))
    w = sets.loc[t4["widest"]]
    t4_line = (
        f"WIDEST SET: {_fmt_set(t4['widest'])} "
        f"(days={int(w['n_days'])}, electrodes={int(w['n_electrodes'])}, stims={int(w['n_stims'])})"
    )
    print(t4_line)

    print("\n=== TASK 5: unstimulated epochs ===")
    t5 = unstimulated_epochs(stim, first_event, last_event, long_gap_h)
    print(f"first event: {first_event}   first stim: {stim['time_of_stim'].min()}")
    print(f"pre-stimulation baseline: {t5['baseline_h']:.2f} h")
    print(f"last stim: {stim['time_of_stim'].max()}   last event: {last_event}")
    print(f"post-stimulation tail: {t5['tail_h']:.2f} h")
    n_long = len(t5["long_gaps"])
    print(f"pooled inter-stim gaps > {long_gap_h:g} h: {n_long} (top {top_gaps}):")
    if n_long:
        print(t5["long_gaps"].head(top_gaps).to_string(index=False))
    t5_line = (
        f"UNSTIMULATED {t5['total_h']:.1f} h: baseline {max(t5['baseline_h'], 0):.1f} h"
        f" + {n_long} gaps > {long_gap_h:g} h ({t5['long_gaps']['duration_h'].sum():.1f} h)"
        f" + tail {max(t5['tail_h'], 0):.1f} h"
    )
    print(f"UNSTIMULATED {t5['total_h']:.1f} h")

    summary = [
        f"{t1['verdict']}: fraction of groups with size == nb_stim_pulse = {t1['match_frac']:.3f}",
        t2_line,
        f"{t3['verdict']}: Spearman rho = {t3['rho']:.3f} (p {_fmt_p(t3['p'])})",
        t4_line,
        t5_line,
    ]
    print("\n=== SUMMARY ===")
    for line in summary:
        print(line)
    return summary


def main() -> None:
    """Parse CLI arguments, load the package read-only, and print the report.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--group-gap-s", type=float, default=DEFAULT_GROUP_GAP_S)
    parser.add_argument("--match-fraction", type=float, default=DEFAULT_MATCH_FRACTION)
    parser.add_argument("--rho-threshold", type=float, default=DEFAULT_RHO_THRESHOLD)
    parser.add_argument("--long-gap-h", type=float, default=DEFAULT_LONG_GAP_H)
    args = parser.parse_args()
    pkg, fs_id = find_package(args.data_dir)
    print(f"Package: {pkg}  (culture: {fs_id}, read-only)\n")
    stim, first_event, last_event = load_tables(pkg, fs_id)
    run_report(
        stim,
        first_event,
        last_event,
        group_gap_s=args.group_gap_s,
        match_fraction=args.match_fraction,
        rho_threshold=args.rho_threshold,
        long_gap_h=args.long_gap_h,
    )


if __name__ == "__main__":
    main()
