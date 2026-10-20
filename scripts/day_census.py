"""Prompt 257 recon: per-day count of 32-block sweeps and preceding non-sweep stimulation in FS369; stdout only."""

import pandas as pd

from gate1_run import H5, STIM_KEY, TUPLE

GAP_S, MIN_SWEEP_BLOCKS, REF_DAY = 15.0, 30, "2025-02-21"
MEDIUM_CHANGE = pd.Timestamp("2025-02-28T14:01:34")
DAY, PRE = pd.Timedelta(days=1), pd.Timedelta(hours=24)

with pd.HDFStore(H5, mode="r") as store:
    stim = store.select(STIM_KEY, columns=["time_of_stim", "electrode", *TUPLE])

# 1. one row per unique timestamp
n_tuples = stim.drop_duplicates(["time_of_stim", *TUPLE]).groupby("time_of_stim").size()
for t in n_tuples[n_tuples > 1].index:
    print(f"NOTE: parameter tuple not constant across electrodes at {t}; carrying first")
u = stim.sort_values("time_of_stim").groupby("time_of_stim", as_index=False).first()
u["tup"] = list(map(tuple, u[TUPLE].to_numpy()))

# 2. blocks: gap > 15 s starts a new block
u["block"] = (u["time_of_stim"].diff().dt.total_seconds() > GAP_S).cumsum()
blocks = u.groupby("block").agg(start=("time_of_stim", "first"), n=("time_of_stim", "size"),
                                tup=("tup", "first"), n_tup=("tup", "nunique"))
for b, r in blocks[blocks["n_tup"] > 1].iterrows():
    print(f"NOTE: block starting {r['start']} has {r['n_tup']} tuples; classifying by first")

# 3. sweep tuple set from the reference day
SWEEP_TUPLES = set(u.loc[u["time_of_stim"].dt.normalize() == pd.Timestamp(REF_DAY), "tup"])
blocks["sweep"] = blocks["tup"].isin(SWEEP_TUPLES)
u["sweep"] = u["block"].map(blocks["sweep"])
nonsweep_t = u.loc[~u["sweep"], "time_of_stim"]

# 4-5. per calendar day
clean, first_sweep = [], {}
days = pd.date_range(u["time_of_stim"].min().normalize(), u["time_of_stim"].max().normalize(), freq="D")
for d in days:
    on_day = u["time_of_stim"].dt.normalize() == d
    b_day = blocks[blocks["start"].dt.normalize() == d]
    sw = b_day[b_day["sweep"]]
    if len(sw):
        t0 = sw["start"].min()
        pre = int(nonsweep_t.between(t0 - PRE, t0, inclusive="left").sum())
        first_sweep[d] = t0
    else:
        t0, pre = None, None
    ok = len(sw) >= MIN_SWEEP_BLOCKS and pre == 0
    if ok:
        clean.append(d)
    fs = t0.strftime("%Y-%m-%dT%H:%M:%S") if t0 is not None else "NA"
    print(f"DAY {d.date()}  stims {int(on_day.sum())}  blocks {len(b_day)}  sweep {len(sw)}  "
          f"first_sweep {fs}  pre24h_nonsweep {'NA' if pre is None else pre}  CLEAN {'yes' if ok else 'no'}")

# sanity checks against known numbers (reported, not enforced)
for d, want_stims, want_blocks in [("2025-02-20", 2360, None), ("2025-02-21", 320, 32)]:
    on_day = u["time_of_stim"].dt.normalize() == pd.Timestamp(d)
    got_stims = int(on_day.sum())
    b_day = blocks[blocks["start"].dt.normalize() == pd.Timestamp(d)]
    if abs(got_stims - want_stims) > 0.05 * want_stims:
        print(f"SANITY MISMATCH: {d} unique stims {got_stims}, expected about {want_stims}")
    if want_blocks and (len(b_day) != want_blocks or set(b_day["n"]) != {want_stims // want_blocks}):
        print(f"SANITY MISMATCH: {d} blocks {len(b_day)} sizes {sorted(set(b_day['n']))}, expected 32 of 10")

# 6. adjacent clean pairs, excluding pairs whose two first sweeps straddle the medium change
pairs = [(a, b) for a, b in zip(clean, clean[1:]) if b - a == DAY
         and not (first_sweep[a] < MEDIUM_CHANGE <= first_sweep[b])]
fmt = lambda p: f"{p[0].date()}/{p[1].date()}"
print(f"SWEEP TUPLES: {len(SWEEP_TUPLES)}")
print(f"CLEAN DAYS: {', '.join(str(d.date()) for d in clean) or 'none'}")
print(f"ADJACENT CLEAN PAIRS: {', '.join(map(fmt, pairs)) or 'none'}")
print(f"MEDIUM CHANGE: {MEDIUM_CHANGE.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"VERDICT: MATCHED PAIR FOUND {', '.join(map(fmt, pairs))}" if pairs else "VERDICT: NO MATCHED PAIR")
