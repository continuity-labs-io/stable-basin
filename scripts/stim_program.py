"""THROWAWAY recon (prompt 255): describe the FS369 stimulation program per day; stdout only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

H5 = Path("data/ephys/finalspark/fs369data/fs369_package.hdf5")
STIM_KEY, DAYS = "fs369_wholelife_stimulations", ["2025-02-20", "2025-02-21"]
TUPLE = ["a1", "a2", "d1", "d2", "nb_stim_pulse", "pulse_train_period", "stim_polarity", "stim_shape"]
CANDIDATE = (0.8, 0.8, 300, 300, 2, 10000, 0, 0)  # in TUPLE order
QUIET_S, MAX_BLOCKS, HALF, TOP_QUIET, CONTEXT = 60.0, 40, 20, 10, 3


def fmt_t(t):
    return pd.Timestamp(t).strftime("%H:%M:%S.%f")[:-3]


def fmt_tup(vals):
    return "(" + ",".join(f"{v:g}" for v in vals) + ")"


def dist(x):
    return f"min={np.min(x):g} median={np.median(x):g} max={np.max(x):g}"


def print_blocks(b, idx, mark=None):
    for i in idx:
        r = b.loc[i]
        print(f"  {'>' if i == mark else ' '}{i:4d} {fmt_tup(r[TUPLE])} n={r.n:4d} {fmt_t(r.start)}-{fmt_t(r.end)}"
              f" within_med_s={r.within:.3f} to_next_s={r.to_next:.3f}")


def describe(day, rows):
    g = rows.groupby("time_of_stim")
    if (g[TUPLE].nunique() > 1).to_numpy().any():
        sys.exit(f"STOP: {day}: a time_of_stim carries more than one parameter tuple")
    ev = g.agg(**{c: (c, "first") for c in TUPLE}, n_elec=("electrode", "nunique")).reset_index()
    ev["gap"] = (ev["time_of_stim"].shift(-1) - ev["time_of_stim"]).dt.total_seconds()
    ev["block"] = (ev[TUPLE] != ev[TUPLE].shift()).any(axis=1).cumsum() - 1
    ev["wgap"] = ev["gap"].where(ev["block"] == ev["block"].shift(-1))
    b = ev.groupby("block").agg(**{c: (c, "first") for c in TUPLE}, n=("gap", "size"),
                                start=("time_of_stim", "first"), end=("time_of_stim", "last"),
                                within=("wgap", "median"))
    b["to_next"] = (b["start"].shift(-1) - b["end"]).dt.total_seconds()
    p = ev.groupby(TUPLE).agg(events=("block", "size"), blocks=("block", "nunique"),
                              first=("time_of_stim", "min"), last=("time_of_stim", "max"))
    print(f"\n=== {day} ===  tuple order: ({','.join(TUPLE)})\nOVERVIEW: events={len(ev)} first={fmt_t(ev.time_of_stim.iloc[0])}"
          f" last={fmt_t(ev.time_of_stim.iloc[-1])} tuples={len(p)} blocks={len(b)} electrodes/event {dist(ev.n_elec)}")
    print("PARAMETER TUPLES:")
    for tup, r in p.sort_values("events", ascending=False).iterrows():
        print(f"  {fmt_tup(tup)} events={r['events']} blocks={r['blocks']} {fmt_t(r['first'])}-{fmt_t(r['last'])}")
    between = ev.gap[ev.wgap.isna() & ev.gap.notna()]
    print(f"BLOCK STRUCTURE: blocks={len(b)} size {dist(b.n)} median within-block gap={ev.wgap.median():.3f}s"
          f" median between-block gap={between.median():.3f}s\nBLOCK SEQUENCE:")
    print_blocks(b, b.index if len(b) <= MAX_BLOCKS else b.index[:HALF].append(b.index[-HALF:]))
    if len(b) > MAX_BLOCKS:
        print(f"  ({len(b) - 2 * HALF} blocks omitted between block {HALF - 1} and block {len(b) - HALF})")
    q = ev[ev.gap > QUIET_S].sort_values("gap", ascending=False)
    print(f"QUIET GAPS > {QUIET_S:g}s: count={len(q)} total={q.gap.sum():.1f}s; top {TOP_QUIET}:")
    for t, s in zip(q.time_of_stim.head(TOP_QUIET), q.gap.head(TOP_QUIET)):
        print(f"  {fmt_t(t)} -> {fmt_t(t + pd.to_timedelta(s, unit='s'))} {s:.1f}s")
    cand = np.logical_and.reduce([np.isclose(b[c], v) for c, v in zip(TUPLE, CANDIDATE)])
    print(f"CANDIDATE IN CONTEXT: {int(cand.sum())} CANDIDATE block(s)")
    for i in b.index[cand]:
        print_blocks(b, range(max(0, i - CONTEXT), min(len(b), i + CONTEXT + 1)), mark=i)
    return dict(events=len(ev), tuples=len(p), blocks=len(b), seq=list(b[TUPLE].itertuples(index=False, name=None)),
                longest=[(s, f"{s:.1f} on {day} starting {fmt_t(q.time_of_stim.iloc[0])}") for s in q.gap.head(1)])


def main():
    stim, res = pd.read_hdf(H5, STIM_KEY), {}
    for day in DAYS:
        rows = stim[stim["time_of_stim"].dt.normalize() == pd.Timestamp(day)]
        if rows.empty or rows[["electrode", "time_of_stim", *TUPLE]].isna().to_numpy().any():
            sys.exit(f"STOP: {day}: no stimulation rows, or NaN in electrode / time_of_stim / parameter tuple")
        res[day] = describe(day, rows)
    s0, s1 = (res[d]["seq"] for d in DAYS)
    div = next((i for i, (x, y) in enumerate(zip(s0, s1)) if x != y), None if len(s0) == len(s1) else min(len(s0), len(s1)))
    print(f"\nDO THE TWO DAYS MATCH: blocks {DAYS[0]}={len(s0)} {DAYS[1]}={len(s1)}; "
          + ("identical block tuple sequences" if div is None else f"first divergence at block {div}"))
    for name, key in (("EVENTS PER DAY", "events"), ("PARAMETER TUPLES", "tuples"), ("BLOCKS", "blocks")):
        print(f"{name}: {res[DAYS[0]][key]}, {res[DAYS[1]][key]}")
    gaps = res[DAYS[0]]["longest"] + res[DAYS[1]]["longest"]
    print("LONGEST QUIET GAP: " + (max(gaps)[1] if gaps else f"none > {QUIET_S:g}s on either day"))
    print("DAYS MATCH" if div is None else f"DAYS DIFFER at block {div}")


if __name__ == "__main__":
    main()
