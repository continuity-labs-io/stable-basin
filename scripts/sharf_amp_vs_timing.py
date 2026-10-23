"""THROWAWAY reconnaissance (prompt 259): amplitude vs timing split-half reliability on the Sharf recordings.

Inventories the Sharf 2022 MaxOne recordings on disk (data/ephys/pharmacological_shock/), finds stimulation
events in the DAC / stimulation-bit auxiliary rows, and stops unless one session has >= 8 trials per stimulus
condition and >= 8 active electrodes. If one does, computes the eight timescale_spike.py measures per
electrode-condition cell, odd/even split-half with condition-centering and Spearman-Brown, and prints
amplitude-vs-timing medians. Prints to stdout only. No tests, config or output files, per prompt 259.
Run from the repository root: python scripts/sharf_amp_vs_timing.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from gate1_run import ACTIVE_HZ, BASELINE_S, CEILING_MIN, POST_HI_S, POST_LO_S, centered_r, latencies
from timescale_spike import BANDS, SPLITS, UNUSABLE_FRAC, all_measures, spearman_brown, undefined

DATA = Path("data/ephys/pharmacological_shock")
FS = 20000.0  # Hz, MaxOne legacy raw.h5 rate; not stored in the file (3,597,600 frames over the 180 s header)
DAC_ROW, BITS_ROW, FRAME_LO_ROW, FRAME_HI_ROW = 1024, 1025, 1026, 1027  # auxiliary rows after the 1024 channels
MIN_TRIALS, MIN_ACTIVE = 8, 8  # prompt 259 hard thresholds, not to be relaxed
KIND = dict(peak_rate="amp", early_amp="amp", hump_amp="amp", late_amp="amp",
            t_return="timing", t_half_peak="timing", t_peak="timing", residual="ratio")
FS369 = dict(amp=0.84, timing=0.34, fast=0.25, slow=0.76)  # quoted from prompt 259 (fs369 timescale_spike run)


def stim_events(f):
    """Find stimulation events as runs of non-resting DAC or non-zero stimulation bits.

    Args:
        f: open h5py File of a legacy MaxOne raw.h5 recording.

    Returns:
        (onsets, amplitude): onset times (s from recording start) and peak |DAC - resting DAC| per event.
    """
    dac, bits = f["sig"][DAC_ROW:BITS_ROW + 1, :].astype(np.int64)
    rest = int(np.bincount(dac).argmax())
    on = np.flatnonzero((dac != rest) | (bits != 0))
    if not on.size:
        return np.empty(0), np.empty(0, dtype=np.int64)
    groups = np.split(on, np.flatnonzero(np.diff(on) > POST_LO_S * FS) + 1)  # frames < POST_LO_S apart: one event
    return np.array([g[0] for g in groups]) / FS, np.array([np.abs(dac[g] - rest).max() for g in groups])


def spikes(f):
    """Load the MaxLab-detected spikes on mapped channels.

    Args:
        f: open h5py File of a legacy MaxOne raw.h5 recording.

    Returns:
        (df, duration): DataFrame with electrode (MaxOne channel id) and t (s from recording start), and the
        recording duration (s).
    """
    frame0 = int(f["sig"][FRAME_HI_ROW, 0]) * 65536 + int(f["sig"][FRAME_LO_ROW, 0])
    st = f["proc0/spikeTimes"][()]
    keep = np.isin(st["channel"], f["mapping"]["channel"])
    return pd.DataFrame(dict(electrode=st["channel"][keep], t=(st["frameno"][keep] - frame0) / FS)), f["sig"].shape[1] / FS


def inventory(path):
    """Print and return one session's stimulation and electrode counts.

    Args:
        path: raw.h5 file.

    Returns:
        dict with session id, onsets, amplitudes, spikes, per-condition trial counts and electrode counts.
    """
    with h5py.File(path, "r") as f:
        onsets, amp = stim_events(f)
        sp, dur = spikes(f)
        n_rec = len(f["mapping"])
    rate = sp["electrode"].value_counts() / dur
    conds, counts = np.unique(amp, return_counts=True)
    s = dict(sid=path.name.removesuffix(".raw.h5"), onsets=onsets, amp=amp, sp=sp, n_rec=n_rec,
             n_active=int((rate > ACTIVE_HZ).sum()), min_trials=int(counts.min()) if counts.size else 0)
    quiet = f"{onsets[0]:.1f} s before first stimulus" if onsets.size else "n/a (no stimulation)"
    print(f"INVENTORY {s['sid']:<18} duration {dur:.1f} s  stim_events {onsets.size}  conditions {conds.size}  "
          f"trials/condition {dict(zip(conds.tolist(), counts.tolist())) or 'n/a'}  recording_electrodes {n_rec}  "
          f"active_electrodes(whole-recording rate > {ACTIVE_HZ} Hz) {s['n_active']}  quiet_pre_stim {quiet}  "
          f"spikes {len(sp)} (outside recording {int(((sp['t'] < 0) | (sp['t'] > dur)).sum())})")
    return s


def fmt(x):
    """Format a median, n/a if undefined.

    Args:
        x: float or NaN.

    Returns:
        two-decimal string or 'n/a'.
    """
    return "n/a" if np.isnan(x) else f"{x:.2f}"


def med(ceilings, names):
    """Median ceiling over the named usable measures.

    Args:
        ceilings: dict measure -> ceiling, None if UNUSABLE.
        names: measures to include.

    Returns:
        float median, NaN if none usable.
    """
    vals = [ceilings[n] for n in names if ceilings[n] is not None and not np.isnan(ceilings[n])]
    return float(np.median(vals)) if vals else float("nan")


def run(s):
    """Split-half reliability of the eight measures for the selected session; prints MEASURE, median and verdicts.

    Args:
        s: inventory dict of the selected session.
    """
    onsets, amp, sp = s["onsets"], s["amp"], s["sp"]
    conds = np.unique(amp)
    blocks = [onsets[amp == c] for c in conds]  # block = all stimuli of one condition
    quiet = [b[0] - max([0.0, *(onsets[onsets < b[0]] + POST_HI_S)]) for b in blocks]
    base_s = min(BASELINE_S, *quiet)
    if base_s <= 0 or any(((onsets >= b[0] - base_s) & (onsets < b[0])).any() for b in blocks):
        sys.exit("STOP: no clean baseline")
    elec = np.sort(sp["electrode"].unique())
    base = np.array([[np.sum((sp["electrode"] == e) & (sp["t"] >= b[0] - base_s) & (sp["t"] < b[0])) / base_s
                      for b in blocks] for e in elec])  # (electrodes, blocks) Hz
    active = elec[(base > ACTIVE_HZ).all(axis=1)]
    if active.size < MIN_ACTIVE:
        sys.exit(f"STOP: no session supports split-half ({active.size} electrodes active in every block baseline)")
    base = pd.DataFrame(base, index=elec)
    ev = pd.DataFrame(dict(electrode=sp["electrode"], time_of_event=pd.to_datetime(sp["t"], unit="s")))
    full, half = [], []
    for (e, k), trials in latencies(ev, active, [pd.to_datetime(b, unit="s").to_numpy() for b in blocks]).items():
        key, bl_hz = dict(electrode=int(e), block=k), float(base.loc[e, k])
        full.append({**key, **all_measures(trials, bl_hz)})
        half += [{**key, "split": sp_, **all_measures(trials[sl], bl_hz)} for sp_, sl in SPLITS.items()]
    full, half = pd.DataFrame(full), pd.DataFrame(half)

    def grid(split, name):
        return half[half["split"] == split].pivot(index="electrode", columns="block", values=name).loc[active].to_numpy()

    print(f"SESSION {s['sid']}  electrodes {active.size}  conditions {conds.size}  "
          f"trials/condition {s['min_trials']}  baseline_s {base_s:.1f}")
    ceilings = {}
    for name, band in BANDS.items():
        ceilings[name] = spearman_brown(centered_r(grid("odd", name), grid("even", name)))
        frac = float(undefined(full, name).mean())
        flag = " UNUSABLE" if frac > UNUSABLE_FRAC else ""
        print(f"MEASURE {name:<13} kind {KIND[name]:<6} band {band:<4}  ceiling {ceilings[name]:.3f}  "
              f"undefined_frac {frac:.2f}{flag}")
        if flag:
            ceilings[name] = None
    kind = {k: med(ceilings, [n for n in KIND if KIND[n] == k]) for k in ("amp", "timing")}
    band = {b: med(ceilings, [n for n in KIND if KIND[n] == "timing" and BANDS[n] == b]) for b in ("fast", "mid", "slow")}
    print(f"KIND MEDIAN CEILING: amp {fmt(kind['amp'])}  timing {fmt(kind['timing'])}")
    print("BAND MEDIAN CEILING (timing only): " + "  ".join(f"{b} {fmt(v)}" for b, v in band.items()))
    print(f"FS369 REFERENCE:     amp {FS369['amp']:.2f}  timing {FS369['timing']:.2f}   "
          f"fast-timing {FS369['fast']:.2f}  slow-timing {FS369['slow']:.2f}")
    beats = kind["amp"] > kind["timing"]  # False if either is NaN
    print(f"VERDICT: {'holds' if beats else 'does not hold'} — amplitude {'beats' if beats else 'does not beat'} timing")
    print(f"VERDICT: fast-band timing {'works' if band['fast'] >= CEILING_MIN else 'fails'} here")


def main():
    print("NOTE: measures (all_measures, gate1_run.measures, EDGES, WIDTHS, WINDOWS, POST_LO_S, POST_HI_S), "
          "centered_r, spearman_brown, BANDS, SPLITS imported unchanged; nothing copied")
    print(f"NOTE: fast-band 'works' means median ceiling >= gate1_run.CEILING_MIN ({CEILING_MIN}); "
          f"active = baseline rate > gate1_run.ACTIVE_HZ ({ACTIVE_HZ} Hz)")
    files = sorted(DATA.glob("*.raw.h5"))
    print(f"FILES in {DATA}: {[p.name for p in files]}; Interval_* files on disk: "
          f"{len([p for p in files if p.name.startswith('Interval_')])}")
    sessions = [inventory(p) for p in files if p.name.startswith("Drug_")]
    ok = [s for s in sessions if s["min_trials"] >= MIN_TRIALS and s["n_active"] >= MIN_ACTIVE]
    if not ok:
        best = max(sessions, key=lambda s: (s["min_trials"], s["n_active"]))
        sys.exit(f"STOP: no session supports split-half (best candidate {best['sid']}: stim_events "
                 f"{best['onsets'].size}, trials/condition {best['min_trials']}, active electrodes {best['n_active']})")
    run(max(ok, key=lambda s: (s["min_trials"], s["n_active"])))


if __name__ == "__main__":
    main()
