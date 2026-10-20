"""THROWAWAY Gate 1 step 1a recon (prompt 254); delete once answered.

Compares pooled event rate 1 s before stimulus 1 vs 1 s before stimuli 2-10.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H5 = Path("data/ephys/finalspark/fs369data/fs369_package.hdf5")
STIM_KEY, EVENT_KEY = "fs369_wholelife_stimulations", "fs369_wholelife_events"
DAYS = ["2025-02-20", "2025-02-21"]
CANDIDATE = dict(a1=0.8, a2=0.8, d1=300, d2=300, nb_stim_pulse=2,
                 pulse_train_period=10000, stim_polarity=0, stim_shape=0)
PRE_S, POST_LO_S, POST_HI_S, N_BINS = 1.0, 0.003, 10.0, 40
RATIO_LO, RATIO_HI = 0.7, 1.4
OUT = Path("output/gate1_1a.png")


def main():
    edges = np.geomspace(POST_LO_S, POST_HI_S, N_BINS + 1)
    centers, widths = np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)
    ratios, curves, pre1_rates = {}, {}, {}
    with pd.HDFStore(H5, mode="r") as store:
        for key in (STIM_KEY, EVENT_KEY):
            if f"/{key}" not in store.keys():
                sys.exit(f"STOP: table {key} missing from {H5}")
        stim = store.get(STIM_KEY)
        missing = [c for c in ["electrode", "time_of_stim", *CANDIDATE] if c not in stim.columns]
        if missing:
            sys.exit(f"STOP: stimulation columns missing: {missing}")
        cand = stim[np.logical_and.reduce([np.isclose(stim[k], v) for k, v in CANDIDATE.items()])]
        for day in DAYS:
            rows = cand[cand["time_of_stim"].dt.strftime("%Y-%m-%d") == day]
            t_stim = rows["time_of_stim"].drop_duplicates().sort_values().reset_index(drop=True)
            electrodes = np.sort(rows["electrode"].unique())
            n_s, n_e = len(t_stim), len(electrodes)
            if n_s < 2:
                sys.exit(f"STOP: {day}: found {n_s} unique CANDIDATE stimuli; need >= 2")
            t0 = t_stim.iloc[0] - pd.to_timedelta(PRE_S, unit="s")
            t1 = t_stim.iloc[-1] + pd.to_timedelta(POST_HI_S, unit="s")
            ev = store.select(EVENT_KEY, where=f"time_of_event >= '{t0}' & time_of_event <= '{t1}'")
            if {"electrode", "time_of_event"} - set(ev.columns):
                sys.exit(f"STOP: events columns missing: {({'electrode', 'time_of_event'} - set(ev.columns))}")
            on_array = ev["electrode"].isin(electrodes)
            t_ev = np.sort(ev.loc[on_array, "time_of_event"].to_numpy())
            # rel[i, j]: seconds from stimulus i to event j.
            rel = (t_ev[None, :] - t_stim.to_numpy()[:, None]) / np.timedelta64(1, "s")
            pre = (rel >= -PRE_S) & (rel < 0)
            n_pre1, n_pre_rest = int(pre[0].sum()), int(pre[1:].sum())
            r1 = n_pre1 / (1 * n_e * PRE_S)
            r_rest = n_pre_rest / ((n_s - 1) * n_e * PRE_S)
            ratio = r_rest / r1 if r1 > 0 else float("inf")
            counts, _ = np.histogram(rel[(rel >= POST_LO_S) & (rel < POST_HI_S)], bins=edges)
            curves[day] = counts / (n_s * n_e * widths)
            ratios[day], pre1_rates[day] = ratio, r1
            print(f"{day}: unique stimuli={n_s} (expected 10), electrodes={n_e}, "
                  f"events off stimulated electrodes excluded={int((~on_array).sum())}")
            print(f"  pre-stim-1 window: {n_pre1} events, rate={r1:.4f} /electrode/s")
            print(f"  pre-stim-2..{n_s} window: {n_pre_rest} events, rate={r_rest:.4f} /electrode/s")
            print(f"  ratio (2..{n_s} / 1) = {ratio:.3f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for day, color in zip(DAYS, ("#1f77b4", "#d62728")):
        ax.step(centers, np.where(curves[day] > 0, curves[day], np.nan), where="mid", color=color, label=f"{day} post-stimulus")
        ax.axhline(pre1_rates[day], color=color, ls="--", lw=1, label=f"{day} 1 s before stimulus 1")
    ax.set(xscale="log", yscale="log", xlabel="time after stimulus (s)",
           ylabel="events / electrode / s (pooled over stimulated electrodes)",
           title="FS369 CANDIDATE stimuli: pooled post-stimulus event rate")
    ax.legend(fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    ok = all(RATIO_LO <= r <= RATIO_HI for r in ratios.values())
    vals = " ".join(f"{d}={r:.3f}" for d, r in ratios.items())
    print("RETURNS WITHIN 10S" if ok else f"DOES NOT RETURN {vals}")


if __name__ == "__main__":
    main()
