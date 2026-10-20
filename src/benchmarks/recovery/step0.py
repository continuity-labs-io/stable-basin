"""Gate 1 Step 0: evidence for the open questions that precede the recovery measurement.

Reads the stimulation, metadata, door-opening and event tables of the FinalSpark
fs369 package file and reports, per task:

- 0a: which UTC calendar days the CANDIDATE parameter set covers, and whether two
  of them are one calendar day apart.
- 0b: a peri-stimulus event-rate histogram (stimulated electrode, other electrodes
  in the same slot, electrodes in other slots) and a proposed blanking window.
- 0c: the distribution of time to the next stimulation and a proposed response
  window.
- 0d: the same, running backwards, and a proposed pre-stimulus baseline window.
- 0e: per-day, per-electrode event rate outside every stimulation's
  [baseline window, response window] span.

Every verdict is a proposal for a human to accept. Nothing is written back into a
config. The events table is read with exactly one ``HDFStore.select`` per CANDIDATE
day, bounded by a ``where`` clause on ``time_of_event``. The raw voltage file is
never opened.

Governing rule: a value that should be read from the file is never replaced by a
default. When a task needs a value that is absent, that task stops and says why.
"""

import argparse
import ast
from dataclasses import dataclass
import datetime as dt
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

logger = logging.getLogger(__name__)

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

# Column names fixed by the FinalSpark package schema (README.md in the data dir).
STIM_TIME_COL = "time_of_stim"
EVENT_TIME_COL = "time_of_event"
ELECTRODE_COL = "electrode"
VOLTAGE_COL = "max_voltage_uv"
DOOR_TIME_COL = "time"
DOOR_OPEN_COL = "incubator_door_opening"

# The three peri-stimulus curves, in category-code order (0, 1, 2).
CURVES: tuple[str, ...] = ("stim_electrode", "same_slot", "other_slots")

REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "input_path",
    "forbidden_input_names",
    "output_dir",
    "tables",
    "metadata_fields",
    "candidate",
    "expected_candidate",
    "slots",
    "event_query_margin_s",
    "psth",
    "blanking",
    "windows",
    "activity",
    "seed",
)

SLOT_ASSUMPTION_NOTE = (
    "UNVERIFIED: slot membership is assumed contiguous in electrode id. The metadata "
    "lists 12 neurospheres across 4 slots, so a slot is not a neurosphere and must "
    "not be treated as a tissue unit downstream."
)


class TaskStopped(Exception):
    """Raised when a task cannot proceed because a value it needs is absent."""


@dataclass
class BlankingProposal:
    """Proposed blanking window for one peri-stimulus curve.

    Args:
        curve: Curve name (one of ``CURVES``).
        pre_stimulus_median_rate_hz: Median rate across pre-stimulus bins, or None.
        threshold_hz: ``pre_stimulus_median_rate_hz * multiplier``, or None.
        blanking_ms: Left edge (ms) of the first post-stimulus bin whose rate is
            below threshold for ``n_consecutive`` bins, or None if undetermined.
        reason: Why ``blanking_ms`` is None; None when it is determined.
        rebound_ms: Left edge of the first bin after the proposed window whose rate
            is back at or above threshold, or None.
        peak_rate_after_hz: Highest rate at or after the proposed window, or None.
        longest_zero_run_ms: (start_ms, end_ms) of the longest run of post-stimulus
            bins with zero events, or None.
        pre_stimulus_mean_rate_hz: Width-weighted mean rate across pre-stimulus
            bins, or None. Reported as evidence; not used by the rule.
    """

    curve: str
    pre_stimulus_median_rate_hz: float | None
    threshold_hz: float | None
    blanking_ms: float | None
    reason: str | None
    rebound_ms: float | None = None
    peak_rate_after_hz: float | None = None
    longest_zero_run_ms: tuple[float, float] | None = None
    pre_stimulus_mean_rate_hz: float | None = None


# ---------------------------------------------------------------------------
# Config and input
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    """Loads the Step 0 YAML config and checks required keys.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed config dictionary.

    Raises:
        KeyError: If a required key is missing.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config {config_path} is missing keys: {missing}")
    return cfg


def check_input_path(input_path: str, forbidden_names: list[str]) -> None:
    """Refuses inputs that are listed as forbidden (e.g. the raw voltage file).

    Args:
        input_path: Path to the package HDF5 file.
        forbidden_names: File names that must never be opened.

    Returns:
        None.

    Raises:
        ValueError: If the file name is forbidden.
        FileNotFoundError: If the file does not exist.
    """
    if Path(input_path).name in set(forbidden_names):
        raise ValueError(f"refusing to open forbidden input {input_path}")
    if not Path(input_path).is_file():
        raise FileNotFoundError(input_path)


def read_small_tables(input_path: str, tables: dict[str, str]) -> dict[str, Any]:
    """Reads the stimulation, metadata and door-opening tables in full.

    Uses ``HDFStore.get`` so that ``HDFStore.select`` is reserved for the events
    table. The events table is not touched here.

    Args:
        input_path: Path to the package HDF5 file.
        tables: Mapping with keys 'stimulations', 'metadata', 'door', 'events'.

    Returns:
        Dict with 'stim' (DataFrame), 'metadata' (DataFrame or None) and 'door'
        (DataFrame or None). Absent optional tables are None.

    Raises:
        KeyError: If the stimulation table is absent.
    """
    out: dict[str, Any] = {}
    with pd.HDFStore(input_path, "r") as store:
        keys = {k.lstrip("/") for k in store.keys()}
        if tables["stimulations"].lstrip("/") not in keys:
            raise KeyError(f"stimulation table {tables['stimulations']} not in {input_path}")
        out["stim"] = store.get(tables["stimulations"])
        out["metadata"] = (
            store.get(tables["metadata"]) if tables["metadata"].lstrip("/") in keys else None
        )
        out["door"] = store.get(tables["door"]) if tables["door"].lstrip("/") in keys else None
        out["has_events"] = tables["events"].lstrip("/") in keys
    return out


def to_naive_utc(times: pd.Series) -> pd.Series:
    """Converts a datetime Series to tz-naive UTC.

    Tz-naive input is taken to be UTC already (the package file stores no zone).

    Args:
        times: Datetime-like Series.

    Returns:
        tz-naive datetime64[ns] Series.
    """
    t = pd.to_datetime(times)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t.astype("datetime64[ns]")


def times_to_ns(times: pd.Series) -> np.ndarray:
    """Converts a datetime Series to int64 nanoseconds since the epoch.

    Args:
        times: Datetime-like Series.

    Returns:
        int64 array.
    """
    return to_naive_utc(times).to_numpy().view(np.int64)


def parse_metadata_list(value: Any) -> list[str]:
    """Parses a metadata cell that holds a list, a list literal, or a scalar.

    Args:
        value: Cell value from the metadata table.

    Returns:
        List of string items.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return [str(v) for v in parsed]
    return [text]


def metadata_value(metadata: pd.DataFrame | None, field: str) -> Any:
    """Reads one field from the one-row metadata table.

    Args:
        metadata: Metadata DataFrame or None.
        field: Column name.

    Returns:
        The cell value.

    Raises:
        TaskStopped: If the table or field is absent or the cell is empty.
    """
    if metadata is None:
        raise TaskStopped("metadata table is absent")
    if field not in metadata.columns:
        raise TaskStopped(f"metadata field '{field}' is absent")
    value = metadata[field].iloc[0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise TaskStopped(f"metadata field '{field}' is empty")
    return value


# ---------------------------------------------------------------------------
# Task 0a
# ---------------------------------------------------------------------------


def filter_candidate(stim: pd.DataFrame, candidate: dict[str, float]) -> pd.DataFrame:
    """Selects stimulations whose parameters equal CANDIDATE exactly.

    Args:
        stim: Full stimulation table.
        candidate: Mapping of column name to required value.

    Returns:
        Matching rows, sorted by time then electrode. The original row index is kept.

    Raises:
        KeyError: If a CANDIDATE column is absent from the table.
    """
    missing = [c for c in candidate if c not in stim.columns]
    if missing:
        raise KeyError(f"stimulation table lacks CANDIDATE columns {missing}")
    mask = np.ones(len(stim), dtype=bool)
    for col, val in candidate.items():
        mask &= stim[col].to_numpy(dtype=float) == float(val)
    out = stim.loc[mask].copy()
    out[STIM_TIME_COL] = to_naive_utc(out[STIM_TIME_COL])
    return out.sort_values([STIM_TIME_COL, ELECTRODE_COL], kind="stable")


def day_table(cand: pd.DataFrame, first_recording: pd.Timestamp | None) -> pd.DataFrame:
    """Summarizes CANDIDATE stimulations per UTC calendar day.

    Args:
        cand: CANDIDATE stimulations with a tz-naive UTC ``time_of_stim`` column.
        first_recording: Date of first recording, or None if unavailable.

    Returns:
        One row per day with columns day, days_since_first_recording, n_stims,
        n_electrodes, first_stim, last_stim.
    """
    days = cand[STIM_TIME_COL].dt.floor("D")
    g = cand.assign(_day=days).groupby("_day")
    out = pd.DataFrame(
        {
            "n_stims": g.size(),
            "n_electrodes": g[ELECTRODE_COL].nunique(),
            "first_stim": g[STIM_TIME_COL].min(),
            "last_stim": g[STIM_TIME_COL].max(),
        }
    ).reset_index(names="day")
    if first_recording is None:
        out.insert(1, "days_since_first_recording", pd.NA)
    else:
        out.insert(1, "days_since_first_recording", (out["day"] - first_recording).dt.days)
    return out


def day_pairs(days_df: pd.DataFrame, medium_changes: list[pd.Timestamp]) -> pd.DataFrame:
    """Lists every pair of CANDIDATE days and flags medium changes between them.

    A medium change is between days a < b when it lies in
    [first stim on day a, last stim on day b]. That interval is deliberately wide:
    a change during either session also disqualifies the pair.

    Args:
        days_df: Output of ``day_table``.
        medium_changes: Medium-change timestamps (tz-naive UTC).

    Returns:
        One row per pair (a < b) with columns day_a, day_b, separation_days,
        adjacent, medium_change_between, disqualified, min_n_electrodes,
        min_n_stims.
    """
    rows = []
    d = days_df.sort_values("day").reset_index(drop=True)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            a, b = d.iloc[i], d.iloc[j]
            sep = int((b["day"] - a["day"]).days)
            between = [m for m in medium_changes if a["first_stim"] <= m <= b["last_stim"]]
            rows.append(
                {
                    "day_a": a["day"],
                    "day_b": b["day"],
                    "separation_days": sep,
                    "adjacent": sep == 1,
                    "medium_change_between": len(between) > 0,
                    "disqualified": len(between) > 0,
                    "min_n_electrodes": int(min(a["n_electrodes"], b["n_electrodes"])),
                    "min_n_stims": int(min(a["n_stims"], b["n_stims"])),
                }
            )
    cols = [
        "day_a",
        "day_b",
        "separation_days",
        "adjacent",
        "medium_change_between",
        "disqualified",
        "min_n_electrodes",
        "min_n_stims",
    ]
    return pd.DataFrame(rows, columns=cols)


def _fmt_day(day: pd.Timestamp) -> str:
    """Formats a day timestamp as YYYY-MM-DD.

    Args:
        day: Day timestamp.

    Returns:
        ISO date string.
    """
    return pd.Timestamp(day).date().isoformat()


def adjacent_pair_verdict(pairs: pd.DataFrame) -> tuple[str, tuple[str, str] | None]:
    """Chooses the best adjacent pair, or reports the closest pair if none.

    Best = most electrodes on the weaker day, then most stimulations on the weaker
    day, then earliest. Disqualified pairs are never chosen as adjacent. When no
    qualified adjacent pair exists, the closest non-disqualified pair is reported
    (or the closest pair of all if every pair is disqualified).

    Args:
        pairs: Output of ``day_pairs``.

    Returns:
        Tuple (verdict line, (day_a, day_b) used downstream or None).
    """
    if pairs.empty:
        return "NO ADJACENT PAIR: fewer than two CANDIDATE days", None
    order = ["min_n_electrodes", "min_n_stims", "day_a"]
    asc = [False, False, True]
    qualified = pairs[pairs["adjacent"] & ~pairs["disqualified"]]
    if not qualified.empty:
        best = qualified.sort_values(order, ascending=asc).iloc[0]
        a, b = _fmt_day(best["day_a"]), _fmt_day(best["day_b"])
        return f"ADJACENT PAIR: {a}, {b}", (a, b)
    pool = pairs[~pairs["disqualified"]]
    if pool.empty:
        pool = pairs
    best = pool.sort_values(["separation_days"] + order, ascending=[True] + asc).iloc[0]
    a, b = _fmt_day(best["day_a"]), _fmt_day(best["day_b"])
    sep = int(best["separation_days"])
    return f"NO ADJACENT PAIR: closest {a}, {b} ({sep} days apart)", (a, b)


def door_openings_in_spans(
    door: pd.DataFrame | None, spans: pd.DataFrame, start_col: str, end_col: str
) -> pd.Series:
    """Counts door openings (True rows) inside each span.

    Args:
        door: Door table with ``time`` and ``incubator_door_opening`` columns.
        spans: DataFrame holding the span bounds.
        start_col: Column of span starts.
        end_col: Column of span ends (inclusive).

    Returns:
        Series of counts aligned with ``spans``.

    Raises:
        TaskStopped: If the door table or its columns are absent.
    """
    if door is None:
        raise TaskStopped("door-opening table is absent")
    for c in (DOOR_TIME_COL, DOOR_OPEN_COL):
        if c not in door.columns:
            raise TaskStopped(f"door-opening table lacks column '{c}'")
    openings = np.sort(times_to_ns(door.loc[door[DOOR_OPEN_COL].astype(bool), DOOR_TIME_COL]))
    starts = times_to_ns(spans[start_col])
    ends = times_to_ns(spans[end_col])
    lo = np.searchsorted(openings, starts, side="left")
    hi = np.searchsorted(openings, ends, side="right")
    return pd.Series(hi - lo, index=spans.index)


# ---------------------------------------------------------------------------
# Task 0b
# ---------------------------------------------------------------------------


def build_bin_edges_ns(
    pre_window_ms: float, post_window_ms: float, post_segments: list[list[float]]
) -> np.ndarray:
    """Builds peri-stimulus bin edges in integer nanoseconds.

    Post-stimulus edges come from contiguous [start, stop, width] segments that
    start at 0 and end at ``post_window_ms``. Pre-stimulus edges are their mirror
    image, clipped to ``-pre_window_ms`` (with that value added as the first edge).

    Args:
        pre_window_ms: Length of the pre-stimulus window in ms.
        post_window_ms: Length of the post-stimulus window in ms.
        post_segments: List of [start_ms, stop_ms, width_ms].

    Returns:
        Strictly increasing int64 array of edges from -pre to +post.

    Raises:
        ValueError: If segments are not contiguous from 0 to ``post_window_ms`` or a
            width does not divide its segment.
    """
    segs = sorted(post_segments, key=lambda s: s[0])
    post_ns = int(round(post_window_ms * NS_PER_MS))
    pre_ns = int(round(pre_window_ms * NS_PER_MS))
    if pre_ns <= 0 or post_ns <= 0:
        raise ValueError("pre_window_ms and post_window_ms must be positive")
    edges: list[int] = []
    expected_start = 0
    for start, stop, width in segs:
        s, e, w = (int(round(x * NS_PER_MS)) for x in (start, stop, width))
        if s != expected_start:
            raise ValueError(f"post segments are not contiguous at {start} ms")
        if w <= 0 or e <= s or (e - s) % w != 0:
            raise ValueError(f"bad segment {[start, stop, width]}")
        edges.extend(range(s, e, w))
        expected_start = e
    if expected_start != post_ns:
        raise ValueError(f"post segments end at {expected_start / NS_PER_MS} ms, not {post_window_ms}")
    post = np.array(edges + [post_ns], dtype=np.int64)
    pre = -post[::-1]
    pre = pre[pre >= -pre_ns]
    if pre[0] != -pre_ns:
        pre = np.concatenate([np.array([-pre_ns], dtype=np.int64), pre])
    return np.unique(np.concatenate([pre, post]))


def slot_lookup(slots: list[list[int]]) -> dict[int, int]:
    """Maps electrode id to slot index from inclusive id ranges.

    Args:
        slots: List of [first_id, last_id] ranges.

    Returns:
        Dict electrode id -> slot index.

    Raises:
        ValueError: If ranges overlap or are reversed.
    """
    out: dict[int, int] = {}
    for k, (lo, hi) in enumerate(slots):
        if hi < lo:
            raise ValueError(f"slot range {[lo, hi]} is reversed")
        for e in range(int(lo), int(hi) + 1):
            if e in out:
                raise ValueError(f"electrode {e} appears in two slots")
            out[e] = k
    return out


def _map_slots(electrodes: np.ndarray, slot_of: dict[int, int]) -> np.ndarray:
    """Vectorized electrode -> slot index lookup; -1 for unmapped electrodes.

    Args:
        electrodes: Integer electrode ids.
        slot_of: Output of ``slot_lookup``.

    Returns:
        int64 array of slot indices, -1 where unmapped.
    """
    keys = np.array(sorted(slot_of), dtype=np.int64)
    vals = np.array([slot_of[k] for k in keys], dtype=np.int64)
    e = np.asarray(electrodes, dtype=np.int64)
    pos = np.clip(np.searchsorted(keys, e), 0, len(keys) - 1)
    found = keys[pos] == e
    return np.where(found, vals[pos], -1)


def collect_peristimulus(
    stim_t_ns: np.ndarray,
    stim_e: np.ndarray,
    ev_t_ns: np.ndarray,
    ev_e: np.ndarray,
    ev_v: np.ndarray,
    pre_ns: int,
    post_ns: int,
    slot_of: dict[int, int],
) -> dict[str, Any]:
    """Collects events in [t - pre, t + post) around each stimulation.

    Each event is labelled 0 (stimulated electrode), 1 (other electrode in the same
    slot) or 2 (electrode in another slot). An event that falls in several windows
    is counted once per window. Events on electrodes outside the slot map are
    dropped and counted.

    Args:
        stim_t_ns: Stimulation times, int64 ns.
        stim_e: Stimulated electrode per stimulation.
        ev_t_ns: Event times, int64 ns, sorted ascending.
        ev_e: Event electrodes.
        ev_v: Event ``max_voltage_uv``.
        pre_ns: Pre-stimulus window length in ns.
        post_ns: Post-stimulus window length in ns.
        slot_of: Electrode -> slot map.

    Returns:
        Dict with 'rel_ns' (int64), 'category' (int8), 'voltage' (float64) and
        'n_unmapped' (int).

    Raises:
        ValueError: If event times are not sorted or a stimulated electrode is not
            in the slot map.
    """
    ev_t_ns = np.asarray(ev_t_ns, dtype=np.int64)
    if ev_t_ns.size > 1 and np.any(np.diff(ev_t_ns) < 0):
        raise ValueError("event times must be sorted ascending")
    stim_t_ns = np.asarray(stim_t_ns, dtype=np.int64)
    stim_slot = _map_slots(stim_e, slot_of)
    if np.any(stim_slot < 0):
        bad = sorted(set(np.asarray(stim_e)[stim_slot < 0].tolist()))
        raise ValueError(f"stimulated electrodes {bad} are not in the slot map")
    lo = np.searchsorted(ev_t_ns, stim_t_ns - pre_ns, side="left")
    hi = np.searchsorted(ev_t_ns, stim_t_ns + post_ns, side="left")
    counts = hi - lo
    total = int(counts.sum())
    stim_idx = np.repeat(np.arange(stim_t_ns.size), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    ev_idx = np.repeat(lo, counts) + (np.arange(total) - starts)
    rel = ev_t_ns[ev_idx] - stim_t_ns[stim_idx]
    e_ev = np.asarray(ev_e, dtype=np.int64)[ev_idx]
    slot_ev = _map_slots(e_ev, slot_of)
    keep = slot_ev >= 0
    same_e = e_ev == np.asarray(stim_e, dtype=np.int64)[stim_idx]
    same_slot = slot_ev == stim_slot[stim_idx]
    category = np.where(same_e, 0, np.where(same_slot, 1, 2)).astype(np.int8)
    return {
        "rel_ns": rel[keep],
        "category": category[keep],
        "voltage": np.asarray(ev_v, dtype=np.float64)[ev_idx][keep],
        "n_unmapped": int((~keep).sum()),
    }


def category_denominators(stim_e: np.ndarray, slot_of: dict[int, int]) -> np.ndarray:
    """Counts electrode-stimulations per curve, the denominator of each rate.

    Args:
        stim_e: Stimulated electrode per stimulation.
        slot_of: Electrode -> slot map.

    Returns:
        int64 array [n_self, n_same_slot, n_other_slots] summed over stimulations.
    """
    sizes: dict[int, int] = {}
    for s in slot_of.values():
        sizes[s] = sizes.get(s, 0) + 1
    total = len(slot_of)
    slots = _map_slots(stim_e, slot_of)
    size_per_stim = np.array([sizes[int(s)] for s in slots], dtype=np.int64)
    return np.array(
        [len(slots), int((size_per_stim - 1).sum()), int((total - size_per_stim).sum())],
        dtype=np.int64,
    )


def psth_table(
    rel_ns: np.ndarray,
    category: np.ndarray,
    voltage: np.ndarray,
    edges_ns: np.ndarray,
    denominators: np.ndarray,
    voltage_percentile: float,
) -> pd.DataFrame:
    """Builds the pooled peri-stimulus histogram, one row per bin.

    Rate is events per electrode per second: count / (electrode-stims * width_s).

    Args:
        rel_ns: Event time relative to its stimulation, int64 ns.
        category: Curve code per event (0, 1, 2).
        voltage: ``max_voltage_uv`` per event.
        edges_ns: Bin edges in ns (from ``build_bin_edges_ns``).
        denominators: Electrode-stimulations per curve.
        voltage_percentile: Upper percentile of voltage to report per bin.

    Returns:
        DataFrame with bin_start_ms, bin_end_ms, bin_width_ms and, per curve,
        count_<c>, rate_hz_<c>, voltage_median_uv_<c>, voltage_p<q>_uv_<c>.
    """
    n_bins = edges_ns.size - 1
    bin_idx = np.searchsorted(edges_ns, rel_ns, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx, category, voltage = bin_idx[valid], category[valid], voltage[valid]
    width_s = np.diff(edges_ns) / NS_PER_S
    out = pd.DataFrame(
        {
            "bin_start_ms": edges_ns[:-1] / NS_PER_MS,
            "bin_end_ms": edges_ns[1:] / NS_PER_MS,
            "bin_width_ms": np.diff(edges_ns) / NS_PER_MS,
        }
    )
    q = float(voltage_percentile)
    q_label = f"{q:g}".replace(".", "_")
    for code, name in enumerate(CURVES):
        sel = category == code
        counts = np.bincount(bin_idx[sel], minlength=n_bins)
        denom = float(denominators[code])
        rate = counts / (denom * width_s) if denom > 0 else np.full(n_bins, np.nan)
        out[f"count_{name}"] = counts
        out[f"rate_hz_{name}"] = rate
        med = np.full(n_bins, np.nan)
        upper = np.full(n_bins, np.nan)
        if sel.any():
            g = pd.Series(voltage[sel]).groupby(bin_idx[sel])
            med_s, up_s = g.median(), g.quantile(q / 100.0)
            med[med_s.index.to_numpy()] = med_s.to_numpy()
            upper[up_s.index.to_numpy()] = up_s.to_numpy()
        out[f"voltage_median_uv_{name}"] = med
        out[f"voltage_p{q_label}_uv_{name}"] = upper
    return out


def propose_blanking(
    curve: str,
    bin_start_ms: np.ndarray,
    bin_end_ms: np.ndarray,
    rate_hz: np.ndarray,
    multiplier: float,
    n_consecutive: int,
) -> BlankingProposal:
    """Finds the first post-stimulus bin whose rate stays below threshold.

    Threshold = median rate over pre-stimulus bins (bin_end <= 0) * multiplier.
    The proposed window is the left edge of the first bin at or after t = 0 whose
    rate, and the rates of the following ``n_consecutive - 1`` bins, are strictly
    below threshold.

    Args:
        curve: Curve name, for labelling.
        bin_start_ms: Bin left edges in ms.
        bin_end_ms: Bin right edges in ms.
        rate_hz: Rate per bin.
        multiplier: Threshold multiplier on the pre-stimulus median rate.
        n_consecutive: Number of consecutive sub-threshold bins required.

    Returns:
        BlankingProposal. ``blanking_ms`` is None, with a reason, if the
        pre-stimulus median is 0 or not finite, or no qualifying run exists.
    """
    bin_start_ms = np.asarray(bin_start_ms, dtype=float)
    rate_hz = np.asarray(rate_hz, dtype=float)
    pre = np.asarray(bin_end_ms, dtype=float) <= 0
    post_idx = np.flatnonzero(bin_start_ms >= 0)
    post_rate = rate_hz[post_idx]
    post_start = bin_start_ms[post_idx]

    zero_run = None
    is_zero = post_rate == 0
    best_len, run_start = 0, None
    for i, z in enumerate(np.append(is_zero, False)):
        if z and run_start is None:
            run_start = i
        elif not z and run_start is not None:
            if i - run_start > best_len:
                best_len = i - run_start
                zero_run = (float(post_start[run_start]), float(bin_end_ms[post_idx[i - 1]]))
            run_start = None

    pre_rates = rate_hz[pre]
    if pre_rates.size == 0 or not np.isfinite(pre_rates).any():
        return BlankingProposal(curve, None, None, None, "no finite pre-stimulus rate", None, None, zero_run)
    pre_widths = (np.asarray(bin_end_ms, dtype=float) - bin_start_ms)[pre]
    pre_mean = float(np.nansum(pre_rates * pre_widths) / pre_widths.sum())
    pre_median = float(np.nanmedian(pre_rates))
    threshold = pre_median * float(multiplier)
    if not threshold > 0:
        reason = (
            f"pre-stimulus median rate is {pre_median:g} Hz at the configured bin widths, "
            "so the threshold is 0 and no bin can fall below it"
        )
        return BlankingProposal(curve, pre_median, threshold, None, reason, None, None, zero_run, pre_mean)

    below = post_rate < threshold
    first = None
    for i in range(len(post_idx) - n_consecutive + 1):
        if below[i : i + n_consecutive].all():
            first = i
            break
    if first is None:
        reason = (
            f"rate never stays below {threshold:g} Hz for {n_consecutive} consecutive "
            "bins within the post-stimulus window"
        )
        return BlankingProposal(curve, pre_median, threshold, None, reason, None, None, zero_run, pre_mean)

    after = post_rate[first:]
    above = np.flatnonzero(~(after < threshold))
    rebound = float(post_start[first + above[0]]) if above.size else None
    peak = float(np.nanmax(after)) if np.isfinite(after).any() else None
    return BlankingProposal(
        curve, pre_median, threshold, float(post_start[first]), None, rebound, peak, zero_run, pre_mean
    )


def coarse_psth(psth: pd.DataFrame, edges_ms: list[float], denominators: np.ndarray) -> pd.DataFrame:
    """Re-aggregates PSTH counts into coarser post-stimulus segments for reading.

    Only PSTH bins lying entirely inside a segment are counted, so segment edges
    should coincide with PSTH bin edges.

    Args:
        psth: Output of ``psth_table``.
        edges_ms: Increasing segment edges in ms.
        denominators: Electrode-stimulations per curve.

    Returns:
        One row per segment with segment_ms and, per curve, count and rate_hz.
    """
    rows = []
    for a, b in zip(edges_ms[:-1], edges_ms[1:]):
        s = psth[(psth["bin_start_ms"] >= a) & (psth["bin_end_ms"] <= b)]
        row: dict[str, Any] = {"segment_ms": f"{a:g}-{b:g}"}
        for code, name in enumerate(CURVES):
            c = int(s[f"count_{name}"].sum())
            row[f"count_{name}"] = c
            d = float(denominators[code])
            row[f"rate_hz_{name}"] = c / (d * (b - a) / 1000.0) if d > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def classify_locality(
    self_ms: float | None, other_ms: float | None, finest_width_ms: float, local_ratio: float
) -> str | None:
    """Labels the artifact LOCAL or GLOBAL from two blanking windows.

    Both values are floored at one finest bin width so that a window of 0 ms is
    treated as "shorter than one bin" rather than an infinite ratio.

    Args:
        self_ms: Blanking window on the stimulated electrode, or None.
        other_ms: Blanking window on other-slot electrodes, or None.
        finest_width_ms: Narrowest bin width in ms.
        local_ratio: Ratio above which the windows count as different.

    Returns:
        'LOCAL', 'GLOBAL', or None if either window is undetermined.
    """
    if self_ms is None or other_ms is None:
        return None
    a, b = max(self_ms, finest_width_ms), max(other_ms, finest_width_ms)
    return "LOCAL" if max(a, b) > local_ratio * min(a, b) else "GLOBAL"


def co_stimulation_counts(cand: pd.DataFrame, stim: pd.DataFrame) -> np.ndarray:
    """Counts distinct electrodes triggered at each CANDIDATE stimulation's timestamp.

    Args:
        cand: CANDIDATE stimulations.
        stim: All stimulations.

    Returns:
        int64 array, one count per CANDIDATE row (1 = no other electrode co-triggered).
    """
    all_t = times_to_ns(stim[STIM_TIME_COL])
    per_t = pd.Series(stim[ELECTRODE_COL].to_numpy()).groupby(all_t).nunique()
    return per_t.reindex(times_to_ns(cand[STIM_TIME_COL])).to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Tasks 0c and 0d
# ---------------------------------------------------------------------------


def compute_gaps(
    cand_t_ns: np.ndarray, cand_e: np.ndarray, all_t_ns: np.ndarray, all_e: np.ndarray
) -> dict[str, np.ndarray]:
    """Computes time to the next and from the previous stimulation, in seconds.

    "Any" uses every stimulation in the table at a strictly different time, so
    triggers sharing the same timestamp do not count as the next one. "Same" uses
    only stimulations on the same electrode. Absent neighbours give ``inf``.

    Args:
        cand_t_ns: CANDIDATE stimulation times, int64 ns.
        cand_e: CANDIDATE electrodes.
        all_t_ns: All stimulation times, int64 ns.
        all_e: All stimulation electrodes.

    Returns:
        Dict with float64 arrays 'gap_before_same', 'gap_before_any',
        'gap_after_same', 'gap_after_any'.
    """
    cand_t_ns = np.asarray(cand_t_ns, dtype=np.int64)
    cand_e = np.asarray(cand_e)
    all_t_ns = np.asarray(all_t_ns, dtype=np.int64)
    all_e = np.asarray(all_e)

    def _neighbours(times: np.ndarray, uniq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nxt = np.searchsorted(uniq, times, side="right")
        prv = np.searchsorted(uniq, times, side="left") - 1
        after = np.full(times.size, np.inf)
        before = np.full(times.size, np.inf)
        ok_n = nxt < uniq.size
        ok_p = prv >= 0
        after[ok_n] = (uniq[nxt[ok_n]] - times[ok_n]) / NS_PER_S
        before[ok_p] = (times[ok_p] - uniq[prv[ok_p]]) / NS_PER_S
        return before, after

    before_any, after_any = _neighbours(cand_t_ns, np.unique(all_t_ns))
    before_same = np.full(cand_t_ns.size, np.inf)
    after_same = np.full(cand_t_ns.size, np.inf)
    for e in np.unique(cand_e):
        rows = cand_e == e
        b, a = _neighbours(cand_t_ns[rows], np.unique(all_t_ns[all_e == e]))
        before_same[rows], after_same[rows] = b, a
    return {
        "gap_before_same": before_same,
        "gap_before_any": before_any,
        "gap_after_same": after_same,
        "gap_after_any": after_any,
    }


def gap_summary(gaps: dict[str, np.ndarray], keys: list[str], percentiles: list[float]) -> pd.DataFrame:
    """Summarizes gap distributions with order-statistic percentiles.

    Uses ``method='lower'`` so that ``inf`` (no neighbour) never interpolates to NaN.

    Args:
        gaps: Output of ``compute_gaps``.
        keys: Which gap arrays to summarize.
        percentiles: Percentiles to report; 0 and 100 are labelled min and max.

    Returns:
        One row per key with columns 'definition', 'n_inf' and one per percentile.
    """
    rows = []
    for k in keys:
        x = gaps[k]
        row: dict[str, Any] = {"definition": k, "n_inf": int(np.isinf(x).sum())}
        for p in percentiles:
            label = "min" if p == 0 else "max" if p == 100 else f"p{p:g}"
            row[label] = float(np.percentile(x, p, method="lower")) if x.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(
    gap_same: np.ndarray,
    gap_any: np.ndarray,
    electrodes: np.ndarray,
    days: np.ndarray,
    lengths_s: list[float],
    min_days: int,
) -> pd.DataFrame:
    """Counts stimulations with at least ``L`` seconds of quiet on both definitions.

    Args:
        gap_same: Same-electrode gap per stimulation (s).
        gap_any: Any-electrode gap per stimulation (s).
        electrodes: Electrode per stimulation.
        days: Day label per stimulation.
        lengths_s: Candidate window lengths in seconds.
        min_days: Days an electrode must survive on to count in
            ``n_electrodes_on_min_days``.

    Returns:
        One row per length with window_s, n_stims, fraction_stims, n_electrodes,
        n_days, n_electrodes_on_min_days.
    """
    df = pd.DataFrame({"e": electrodes, "d": days})
    n = len(df)
    rows = []
    for L in lengths_s:
        ok = (gap_same >= L) & (gap_any >= L)
        sub = df[ok]
        per_e = sub.groupby("e")["d"].nunique()
        rows.append(
            {
                "window_s": float(L),
                "n_stims": int(ok.sum()),
                "fraction_stims": float(ok.sum() / n) if n else np.nan,
                "n_electrodes": int(sub["e"].nunique()),
                "n_days": int(sub["d"].nunique()),
                "n_electrodes_on_min_days": int((per_e >= min_days).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_window(cov: pd.DataFrame, n_required_electrodes: int, min_fraction: float) -> float | None:
    """Largest window keeping ``min_fraction`` of stims and every electrode on enough days.

    Args:
        cov: Output of ``coverage_table``.
        n_required_electrodes: Electrodes that must all survive on ``min_days``.
        min_fraction: Minimum fraction of stimulations retained.

    Returns:
        Window length in seconds, or None if no length satisfies both constraints.
    """
    ok = cov[
        (cov["fraction_stims"] >= min_fraction)
        & (cov["n_electrodes_on_min_days"] >= n_required_electrodes)
    ]
    return None if ok.empty else float(ok["window_s"].max())


# ---------------------------------------------------------------------------
# Task 0e
# ---------------------------------------------------------------------------


def exclusion_intervals(
    stim_t_ns: np.ndarray, before_ns: int, after_ns: int, span: tuple[int, int]
) -> np.ndarray:
    """Merges [t - before, t + after) around every stimulation, clipped to ``span``.

    Args:
        stim_t_ns: Stimulation times, int64 ns.
        before_ns: Pre-stimulus exclusion in ns.
        after_ns: Post-stimulus exclusion in ns.
        span: (start_ns, end_ns) of the observed span.

    Returns:
        int64 array of shape (n, 2), disjoint, sorted, half-open intervals.
    """
    t = np.unique(np.asarray(stim_t_ns, dtype=np.int64))
    s, e = t - before_ns, t + after_ns
    keep = (e > span[0]) & (s < span[1])
    s, e = np.clip(s[keep], span[0], span[1]), np.clip(e[keep], span[0], span[1])
    merged: list[list[int]] = []
    for a, b in zip(s.tolist(), e.tolist()):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.array(merged, dtype=np.int64).reshape(-1, 2)


def quiet_rates(
    ev_t_ns: np.ndarray,
    ev_e: np.ndarray,
    electrodes: list[int],
    span: tuple[int, int],
    excluded: np.ndarray,
) -> tuple[pd.Series, float]:
    """Event rate per electrode inside ``span`` but outside the excluded intervals.

    Args:
        ev_t_ns: Event times, int64 ns, sorted.
        ev_e: Event electrodes.
        electrodes: Electrodes to report (zero-event electrodes get 0 Hz).
        span: (start_ns, end_ns), inclusive of both ends for event membership.
        excluded: Output of ``exclusion_intervals``.

    Returns:
        Tuple (rate in Hz per electrode, quiet duration in seconds).

    Raises:
        TaskStopped: If the quiet duration is not positive.
    """
    ev_t_ns = np.asarray(ev_t_ns, dtype=np.int64)
    quiet_ns = (span[1] - span[0]) - int((excluded[:, 1] - excluded[:, 0]).sum())
    if quiet_ns <= 0:
        raise TaskStopped("no quiet time inside the loaded span")
    in_span = (ev_t_ns >= span[0]) & (ev_t_ns <= span[1])
    if excluded.shape[0]:
        pos = np.searchsorted(excluded[:, 0], ev_t_ns, side="right") - 1
        inside = (pos >= 0) & (ev_t_ns < excluded[np.clip(pos, 0, None), 1])
    else:
        inside = np.zeros(ev_t_ns.size, dtype=bool)
    sel = in_span & ~inside
    counts = pd.Series(np.asarray(ev_e)[sel]).value_counts()
    quiet_s = quiet_ns / NS_PER_S
    rates = counts.reindex(electrodes, fill_value=0).astype(float) / quiet_s
    rates.index.name = ELECTRODE_COL
    return rates, quiet_s


def longest_event_gap_s(ev_t_ns: np.ndarray, span: tuple[int, int]) -> float:
    """Longest stretch inside ``span`` with no event on any electrode.

    Args:
        ev_t_ns: Event times, int64 ns, sorted.
        span: (start_ns, end_ns).

    Returns:
        Gap length in seconds (the whole span if there are no events).
    """
    pts = np.concatenate([[span[0]], np.asarray(ev_t_ns, dtype=np.int64), [span[1]]])
    return float(np.diff(pts).max() / NS_PER_S)


# ---------------------------------------------------------------------------
# Events access
# ---------------------------------------------------------------------------


def query_events(store: pd.HDFStore, key: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Reads events with ``t0 <= time_of_event <= t1`` in one bounded select.

    Args:
        store: Open HDFStore.
        key: Events table key.
        t0: Inclusive lower bound (tz-naive UTC).
        t1: Inclusive upper bound (tz-naive UTC).

    Returns:
        Events DataFrame for the span.
    """
    where = f"{EVENT_TIME_COL} >= '{t0}' & {EVENT_TIME_COL} <= '{t1}'"
    return store.select(key, where=where)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def code_version() -> dict[str, Any]:
    """Reads the git commit and dirty flag of the working tree.

    Returns:
        Dict with 'git_commit' and 'git_dirty'; values are None if git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """Renders a DataFrame as a GitHub markdown table.

    Args:
        df: Table to render.
        floatfmt: Float format for numeric cells.

    Returns:
        Markdown string.
    """
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy, pandas and datetime values.

    Args:
        o: Object to encode.

    Returns:
        A JSON-serializable equivalent.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else str(f)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is pd.NA:
        return None
    return str(o)


def plot_artifact_psth(csv_path: str, png_path: str, proposals: dict[str, BlankingProposal]) -> None:
    """Plots the post-stimulus PSTH from the CSV table on a log time axis.

    Args:
        csv_path: Path to ``artifact_psth.csv`` (the only data source for the plot).
        png_path: Output PNG path.
        proposals: Blanking proposals per curve, for threshold and window lines.

    Returns:
        None.
    """
    tab = pd.read_csv(csv_path)
    post = tab[tab["bin_start_ms"] >= 0].copy()
    x = 0.5 * (post["bin_start_ms"] + post["bin_end_ms"])
    p95_cols = [c for c in tab.columns if c.startswith("voltage_p") and c.endswith(CURVES[0])]
    colors = {"stim_electrode": "#d1495b", "same_slot": "#edae49", "other_slots": "#00798c"}
    styles = {"stim_electrode": ("-", 2.4), "same_slot": ("--", 1.4), "other_slots": (":", 1.0)}
    fig, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name in CURVES:
        ls, lw = styles[name]
        ax_r.step(x, post[f"rate_hz_{name}"], where="mid", color=colors[name], label=name, ls=ls, lw=lw)
        prop = proposals.get(name)
        if prop is not None and prop.threshold_hz:
            ax_r.axhline(prop.threshold_hz, color=colors[name], ls=":", lw=0.8)
        if prop is not None and prop.blanking_ms is not None and prop.blanking_ms > 0:
            ax_r.axvline(prop.blanking_ms, color=colors[name], ls="--", lw=1.2,
                         label=f"proposed blanking ({name}) {prop.blanking_ms:g} ms")
    ref = proposals.get(CURVES[0])
    if ref is not None and ref.pre_stimulus_mean_rate_hz is not None:
        ax_r.axhline(ref.pre_stimulus_mean_rate_hz, color="#888", ls="-.", lw=0.8,
                     label=f"pre-stimulus mean rate ({CURVES[0]})")
    ax_r.set_xscale("log")
    ax_r.set_yscale("symlog", linthresh=1.0)
    ax_r.set_ylim(bottom=0)
    ax_r.set_ylabel("events per electrode per s (pooled)")
    ax_r.set_title("Peri-stimulus event rate, CANDIDATE stimulations (thin dotted: threshold, if > 0)")
    ax_r.legend(fontsize=7)
    ax_v.step(x, post[f"voltage_median_uv_{CURVES[0]}"], where="mid", color="#444", label="median")
    if p95_cols:
        ax_v.step(x, post[p95_cols[0]], where="mid", color=colors[CURVES[0]], label=p95_cols[0])
    ax_v.set_yscale("symlog", linthresh=100.0)
    ax_v.set_xlabel("time since stimulation (ms, bin centre, log scale)")
    ax_v.set_ylabel("max_voltage_uv, stimulated electrode")
    ax_v.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_step0(config_path: str = "configs/recovery_gate1.yaml") -> dict[str, Any]:
    """Runs Step 0 end to end and writes every output file.

    Args:
        config_path: Path to the Step 0 YAML config.

    Returns:
        The sidecar dictionary written to ``step0.json``.
    """
    cfg = load_config(config_path)
    input_path = cfg["input_path"]
    check_input_path(input_path, cfg["forbidden_input_names"])
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    tables = cfg["tables"]
    psth_cfg, blank_cfg, win_cfg, act_cfg = cfg["psth"], cfg["blanking"], cfg["windows"], cfg["activity"]
    slot_of = slot_lookup(cfg["slots"])
    margin_ns = int(round(float(cfg["event_query_margin_s"]) * NS_PER_S))
    pre_ns = int(round(float(psth_cfg["pre_window_ms"]) * NS_PER_MS))
    post_ns = int(round(float(psth_cfg["post_window_ms"]) * NS_PER_MS))
    if margin_ns < max(pre_ns, post_ns):
        raise ValueError("event_query_margin_s must cover the peri-stimulus window")
    edges_ns = build_bin_edges_ns(psth_cfg["pre_window_ms"], psth_cfg["post_window_ms"], psth_cfg["post_segments"])

    small = read_small_tables(input_path, tables)
    stim = small["stim"].copy()
    stim[STIM_TIME_COL] = to_naive_utc(stim[STIM_TIME_COL])
    metadata, door = small["metadata"], small["door"]
    report: list[str] = ["# Gate 1 Step 0 report", ""]
    report += [
        f"Input: `{input_path}`. Config: `{config_path}`. Generated by "
        "`src/benchmarks/recovery/step0.py`. Every verdict below is a proposal for a "
        "human to accept; none is written into a config.",
        "",
        "Timestamps in the file carry no time zone and are treated as UTC.",
        "",
    ]
    sidecar: dict[str, Any] = {
        "script": "src/benchmarks/recovery/step0.py",
        "config_path": config_path,
        "parameters": cfg,
        "code_version": code_version(),
        "seed": cfg["seed"],
        "seed_note": "Step 0 performs no random operation; the seed is recorded per the output contract.",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_file": {"path": input_path, "bytes": os.path.getsize(input_path)},
        "timezone_assumption": "time columns are tz-naive in the file; treated as UTC",
        "slot_assumption": {"slots": cfg["slots"], "status": "UNVERIFIED", "note": SLOT_ASSUMPTION_NOTE},
    }
    verdicts: dict[str, str] = {}

    # ---- CANDIDATE filter ---------------------------------------------------
    cand = filter_candidate(stim, cfg["candidate"])
    cand_days = cand[STIM_TIME_COL].dt.floor("D")
    exp = cfg["expected_candidate"]
    observed = {
        "n_stims": int(len(cand)),
        "n_electrodes": int(cand[ELECTRODE_COL].nunique()),
        "n_days": int(cand_days.nunique()),
    }
    count_diff = {k: {"expected": exp[k], "observed": observed[k]} for k in exp if exp[k] != observed[k]}
    sidecar["candidate_counts"] = {"expected": exp, "observed": observed, "differences": count_diff}
    report += ["## CANDIDATE selection", ""]
    report.append(
        f"Observed {observed['n_stims']} stimulations, {observed['n_electrodes']} electrodes, "
        f"{observed['n_days']} days (expected {exp['n_stims']}, {exp['n_electrodes']}, {exp['n_days']})."
    )
    report.append(
        "**Count differs from expectation:** " + json.dumps(count_diff) if count_diff
        else "Counts match the design doc."
    )
    report.append("")
    cand_electrodes = sorted(int(e) for e in cand[ELECTRODE_COL].unique())
    n_req_e = len(cand_electrodes)

    co = co_stimulation_counts(cand, stim)
    co_summary = {"min": int(co.min()), "median": float(np.median(co)), "max": int(co.max())} if co.size else {}
    sidecar["co_stimulation_electrodes_per_timestamp"] = co_summary
    report.append(
        f"Distinct electrodes triggered at the same timestamp as each CANDIDATE stimulation "
        f"(all parameter sets): min {co_summary.get('min')}, median {co_summary.get('median')}, "
        f"max {co_summary.get('max')}."
    )
    report.append("")

    # ---- Task 0a ----------------------------------------------------------
    report += ["## Task 0a: CANDIDATE days and adjacent pairs", ""]
    pair_for_0e: tuple[str, str] | None = None
    days_df = pd.DataFrame()
    try:
        first_rec = pd.Timestamp(str(metadata_value(metadata, cfg["metadata_fields"]["first_recording_date"])))
        mc_raw = metadata_value(metadata, cfg["metadata_fields"]["medium_changes"])
        medium_changes = [pd.Timestamp(s) for s in parse_metadata_list(mc_raw)]
        days_df = day_table(cand, first_rec.normalize())
        days_df["door_openings_in_stim_span"] = door_openings_in_spans(door, days_df, "first_stim", "last_stim")
        days_df["medium_change_same_day"] = [
            ", ".join(m.isoformat() for m in medium_changes if m.normalize() == d) or "none"
            for d in days_df["day"]
        ]
        pairs = day_pairs(days_df, medium_changes)
        verdicts["0a"], pair_for_0e = adjacent_pair_verdict(pairs)
        show_days = days_df.assign(day=days_df["day"].map(_fmt_day))
        report += [_md(show_days), ""]
        report.append(f"Days: {', '.join(show_days['day'])}.")
        report.append(f"First recording (metadata `Date`): {first_rec.date().isoformat()}.")
        report.append(f"Medium changes (metadata): {', '.join(m.isoformat() for m in medium_changes)}.")
        adj = pairs[pairs["adjacent"]]
        report.append("")
        report.append("Pairs exactly one calendar day apart:")
        report.append("")
        report.append(
            _md(adj.assign(day_a=adj["day_a"].map(_fmt_day), day_b=adj["day_b"].map(_fmt_day)))
            if not adj.empty else "none"
        )
        report.append("")
        report.append(
            "Medium change between any pair (window = first stim of earlier day to last stim of later day): "
            + ("; ".join(
                f"{_fmt_day(r.day_a)}/{_fmt_day(r.day_b)}" for r in pairs.itertuples() if r.medium_change_between
            ) or "none")
            + "."
        )
        n_door = int(days_df["door_openings_in_stim_span"].sum())
        report.append(f"Door openings inside any day's CANDIDATE stimulation span: {n_door}.")
        report.append("")
        sidecar["task_0a"] = {
            "days": show_days.to_dict(orient="records"),
            "pairs": pairs.assign(day_a=pairs["day_a"].map(_fmt_day), day_b=pairs["day_b"].map(_fmt_day)).to_dict(orient="records"),
            "medium_changes": [m.isoformat() for m in medium_changes],
            "pair_used_for_0e": pair_for_0e,
        }
    except TaskStopped as exc:
        verdicts["0a"] = f"ADJACENT PAIR STOPPED: {exc}"
        report += [f"STOPPED: {exc}", ""]
        sidecar["task_0a"] = {"stopped": str(exc)}

    # ---- Tasks 0c / 0d (stimulation tables only) ----------------------------
    cd_lines: list[str] = []
    all_t = times_to_ns(stim[STIM_TIME_COL])
    all_e = stim[ELECTRODE_COL].to_numpy()
    cand_t = times_to_ns(cand[STIM_TIME_COL])
    cand_e = cand[ELECTRODE_COL].to_numpy()
    gaps = compute_gaps(cand_t, cand_e, all_t, all_e)
    day_labels = cand_days.map(_fmt_day).to_numpy()
    lengths = [float(x) for x in win_cfg["candidate_lengths_s"]]
    min_days = int(win_cfg["min_days_per_electrode"])
    min_frac = float(win_cfg["min_fraction_stims"])

    cd_lines += ["## Task 0c: usable response window", ""]
    cd_lines.append(
        "Gaps use every stimulation in the table (any parameter set). 'any' ignores "
        "triggers sharing the same timestamp; 'inf' means no later stimulation exists."
    )
    cd_lines.append("")
    cd_lines += [_md(gap_summary(gaps, ["gap_after_same", "gap_after_any"], win_cfg["gap_percentiles"])), ""]
    cov_after = coverage_table(gaps["gap_after_same"], gaps["gap_after_any"], cand_e, day_labels, lengths, min_days)
    cd_lines += [_md(cov_after), ""]
    response_s = choose_window(cov_after, n_req_e, min_frac)
    verdicts["0c"] = (
        f"RESPONSE WINDOW {response_s:g} s" if response_s is not None
        else "NO WINDOW SATISFIES CONSTRAINTS (response)"
    )
    cd_lines.append(
        f"Rule: largest length keeping >= {min_frac:.0%} of stimulations and all {n_req_e} "
        f"electrodes on >= {min_days} days."
    )
    cd_lines.append("")

    cd_lines += ["## Task 0d: pre-stimulus baseline window", ""]
    cd_lines += [_md(gap_summary(gaps, ["gap_before_same", "gap_before_any"], win_cfg["gap_percentiles"])), ""]
    cov_before = coverage_table(gaps["gap_before_same"], gaps["gap_before_any"], cand_e, day_labels, lengths, min_days)
    cd_lines += [_md(cov_before), ""]
    baseline_s = choose_window(cov_before, n_req_e, min_frac)
    verdicts["0d"] = (
        f"BASELINE WINDOW {baseline_s:g} s" if baseline_s is not None
        else "NO WINDOW SATISFIES CONSTRAINTS (baseline)"
    )

    if response_s is not None and baseline_s is not None:
        usable = (
            (gaps["gap_before_same"] >= baseline_s)
            & (gaps["gap_before_any"] >= baseline_s)
            & (gaps["gap_after_same"] >= response_s)
            & (gaps["gap_after_any"] >= response_s)
        )
    else:
        usable = np.zeros(len(cand), dtype=bool)
    windows_df = pd.DataFrame(
        {
            "stim_id": cand.index.to_numpy(),
            "electrode": cand_e,
            "time_of_stim": cand[STIM_TIME_COL].map(lambda t: t.isoformat()).to_numpy(),
            "day": day_labels,
            **gaps,
            "usable": usable,
        }
    )
    windows_df = windows_df[
        ["stim_id", "electrode", "time_of_stim", "day", "gap_before_same", "gap_before_any",
         "gap_after_same", "gap_after_any", "usable"]
    ]
    windows_df.to_csv(os.path.join(out_dir, "stim_windows.csv"), index=False)
    by_day = windows_df.groupby("day").agg(n_stims=("usable", "size"), n_usable=("usable", "sum")).reset_index()
    by_e = windows_df.groupby("electrode").agg(n_stims=("usable", "size"), n_usable=("usable", "sum")).reset_index()
    cd_lines.append(
        f"Joint count (both windows satisfied): {int(usable.sum())} of {len(cand)} CANDIDATE stimulations."
    )
    cd_lines += ["", "By day:", "", _md(by_day), "", "By electrode:", "", _md(by_e), ""]
    sidecar["task_0c"] = {"coverage": cov_after.to_dict(orient="records"), "proposed_s": response_s}
    sidecar["task_0d"] = {
        "coverage": cov_before.to_dict(orient="records"),
        "proposed_s": baseline_s,
        "n_usable_joint": int(usable.sum()),
    }

    # ---- Per-day event pass (Tasks 0b and 0e) -------------------------------
    rel_parts: list[np.ndarray] = []
    cat_parts: list[np.ndarray] = []
    volt_parts: list[np.ndarray] = []
    n_unmapped = 0
    per_day_rows: dict[str, int] = {}
    per_day_spans: dict[str, list[str]] = {}
    rates_by_day: dict[str, pd.Series] = {}
    quiet_by_day: dict[str, float] = {}
    max_gap_by_day: dict[str, float] = {}
    stop_0e: str | None = None
    if response_s is None or baseline_s is None:
        stop_0e = "0c or 0d proposed no window, so the quiet periods are undefined"
    if not small["has_events"]:
        raise KeyError(f"events table {tables['events']} not in {input_path}")

    with pd.HDFStore(input_path, "r") as store:
        for day in sorted(cand_days.unique()):
            label = _fmt_day(day)
            rows = cand_days == day
            d_t = cand_t[rows.to_numpy()]
            d_e = cand_e[rows.to_numpy()]
            t0 = pd.Timestamp(int(d_t.min() - margin_ns), unit="ns")
            t1 = pd.Timestamp(int(d_t.max() + margin_ns), unit="ns")
            ev = query_events(store, tables["events"], t0, t1)
            per_day_rows[label] = int(len(ev))
            per_day_spans[label] = [t0.isoformat(), t1.isoformat()]
            logger.info("Day %s: %d events in [%s, %s]", label, len(ev), t0, t1)
            ev_t = times_to_ns(ev[EVENT_TIME_COL])
            order = np.argsort(ev_t, kind="stable")
            ev_t = ev_t[order]
            ev_e = ev[ELECTRODE_COL].to_numpy()[order]
            ev_v = ev[VOLTAGE_COL].to_numpy()[order]
            del ev
            got = collect_peristimulus(d_t, d_e, ev_t, ev_e, ev_v, pre_ns, post_ns, slot_of)
            rel_parts.append(got["rel_ns"])
            cat_parts.append(got["category"])
            volt_parts.append(got["voltage"])
            n_unmapped += got["n_unmapped"]
            span = (int(t0.value), int(t1.value))
            max_gap_by_day[label] = longest_event_gap_s(ev_t, span)
            if stop_0e is None:
                excl = exclusion_intervals(
                    all_t, int(round(baseline_s * NS_PER_S)), int(round(response_s * NS_PER_S)), span
                )
                try:
                    rates_by_day[label], quiet_by_day[label] = quiet_rates(ev_t, ev_e, cand_electrodes, span, excl)
                except TaskStopped as exc:
                    stop_0e = f"{label}: {exc}"
            del ev_t, ev_e, ev_v

    sidecar["events_query"] = {
        "n_selects": len(per_day_rows),
        "rows_per_day": per_day_rows,
        "span_per_day": per_day_spans,
        "margin_s": cfg["event_query_margin_s"],
        "n_peristimulus_events_on_unmapped_electrodes": n_unmapped,
    }

    # ---- Task 0b ------------------------------------------------------------
    report += ["## Task 0b: stimulation artifact and blanking window", ""]
    denom = category_denominators(cand_e, slot_of)
    rel_all = np.concatenate(rel_parts) if rel_parts else np.array([], np.int64)
    cat_all = np.concatenate(cat_parts) if cat_parts else np.array([], np.int8)
    volt_all = np.concatenate(volt_parts) if volt_parts else np.array([], float)
    psth = psth_table(rel_all, cat_all, volt_all, edges_ns, denom, float(psth_cfg["voltage_percentile"]))
    csv_path = os.path.join(out_dir, "artifact_psth.csv")
    psth.to_csv(csv_path, index=False)
    mult = float(blank_cfg["threshold_multiplier"])
    n_consec = int(blank_cfg["n_consecutive_bins"])
    proposals = {
        name: propose_blanking(
            name, psth["bin_start_ms"].to_numpy(), psth["bin_end_ms"].to_numpy(),
            psth[f"rate_hz_{name}"].to_numpy(), mult, n_consec,
        )
        for name in CURVES
    }
    finest = float(psth["bin_width_ms"].min())
    locality = classify_locality(
        proposals["stim_electrode"].blanking_ms, proposals["other_slots"].blanking_ms,
        finest, float(blank_cfg["local_ratio"]),
    )
    plot_artifact_psth(csv_path, os.path.join(out_dir, "artifact_psth.png"), proposals)

    report.append(
        f"Electrode-stimulations per curve (rate denominators): "
        + ", ".join(f"{n}={int(d)}" for n, d in zip(CURVES, denom))
        + f". Peri-stimulus events on electrodes outside the slot map: {n_unmapped}."
    )
    report.append(f"Slot map: {cfg['slots']}. {SLOT_ASSUMPTION_NOTE}")
    report.append("")
    if co_summary and co_summary["median"] > 1:
        report.append(
            f"> **WARNING:** CANDIDATE stimulations are co-triggered with a median of "
            f"{co_summary['median']:g} electrodes at the identical timestamp. 'Same slot' and "
            "'other slots' electrodes are themselves being stimulated at that instant, so the "
            "three-way split does not separate a local artifact from a global one."
        )
        report.append("")
    prop_rows = []
    for name, p in proposals.items():
        prop_rows.append(
            {
                "curve": name,
                "pre_median_rate_hz": p.pre_stimulus_median_rate_hz,
                "pre_mean_rate_hz": p.pre_stimulus_mean_rate_hz,
                "threshold_hz": p.threshold_hz,
                "proposed_blanking_ms": p.blanking_ms,
                "rebound_at_ms": p.rebound_ms,
                "peak_rate_after_window_hz": p.peak_rate_after_hz,
                "longest_zero_run_ms": None if p.longest_zero_run_ms is None
                else f"{p.longest_zero_run_ms[0]:g}-{p.longest_zero_run_ms[1]:g}",
                "reason_if_undetermined": p.reason,
            }
        )
    report += [_md(pd.DataFrame(prop_rows)), ""]
    report.append(
        f"Rule: first bin at or after 0 ms whose rate is below pre-stimulus median x {mult:g} "
        f"for {n_consec} consecutive bins. 'rebound_at_ms' is the first later bin back at or "
        "above threshold; a rebound means the rate leaves and re-enters the threshold band "
        "after the proposed window."
    )
    s_ms, o_ms = proposals["stim_electrode"].blanking_ms, proposals["other_slots"].blanking_ms
    if locality == "LOCAL":
        report.append("")
        report.append(
            f"> **WARNING: stimulated-electrode window ({s_ms:g} ms) and other-slot window "
            f"({o_ms:g} ms) differ by more than x{float(blank_cfg['local_ratio']):g}. The artifact "
            "looks local; a single global blanking window would be wrong.**"
        )
    report.append("")
    coarse = coarse_psth(psth, [float(x) for x in psth_cfg["report_summary_edges_ms"]], denom)
    report += [
        "Coarse view (bins pooled into reading segments; full table in `artifact_psth.csv`):",
        "",
        _md(coarse),
        "",
    ]
    if s_ms is not None:
        verdicts["0b"] = f"BLANKING {s_ms:g} ms {locality}" if locality else f"BLANKING {s_ms:g} ms (locality undetermined: {proposals['other_slots'].reason})"
    else:
        verdicts["0b"] = f"BLANKING UNDETERMINED: {proposals['stim_electrode'].reason}"
    sidecar["task_0b"] = {
        "bin_edges_ms": (edges_ns / NS_PER_MS).tolist(),
        "pre_edges_note": "pre-stimulus edges mirror the post-stimulus edges, clipped to pre_window_ms",
        "rate_denominators": dict(zip(CURVES, denom.tolist())),
        "proposals": {k: vars(v) for k, v in proposals.items()},
        "locality": locality,
        "locality_floor_ms": finest,
        "coarse_psth": coarse.to_dict(orient="records"),
    }

    report += cd_lines

    # ---- Task 0e ------------------------------------------------------------
    report += ["## Task 0e: quiet-period activity", ""]
    thresholds = [float(x) for x in act_cfg["rate_thresholds_hz"]]
    v_thr = float(act_cfg["verdict_threshold_hz"])
    if stop_0e is None and pair_for_0e is None:
        stop_0e = "0a produced no day pair"
    if stop_0e is None:
        rate_tab = pd.DataFrame(rates_by_day)
        report.append(
            f"Quiet time = loaded span (CANDIDATE span +/- {cfg['event_query_margin_s']:g} s) minus "
            f"[t - {baseline_s:g} s, t + {response_s:g} s) around every stimulation of any parameter set. "
            "Recording continuity inside the span is not known from the package file; the longest "
            "stretch with no event on any electrode is listed as a check."
        )
        report.append("")
        summ = pd.DataFrame(
            {
                "day": list(rates_by_day),
                "quiet_s": [quiet_by_day[d] for d in rates_by_day],
                "longest_no_event_gap_s": [max_gap_by_day[d] for d in rates_by_day],
                **{f"n_electrodes_gt_{t:g}_hz": [int((rates_by_day[d] > t).sum()) for d in rates_by_day] for t in thresholds},
            }
        )
        report += [_md(summ), "", "Rate (Hz) by electrode and day:", "", _md(rate_tab.reset_index()), ""]
        a, b = pair_for_0e
        if a in rate_tab.columns and b in rate_tab.columns:
            active = int(((rate_tab[a] > v_thr) & (rate_tab[b] > v_thr)).sum())
            verdicts["0e"] = f"ACTIVE ELECTRODES {active} of {n_req_e}"
            report.append(f"Active on both {a} and {b} at > {v_thr:g} Hz: {active} of {n_req_e}.")
            report.append("")
        else:
            stop_0e = f"pair {a}/{b} not among loaded days"
        sidecar["task_0e"] = {
            "quiet_s": quiet_by_day,
            "longest_no_event_gap_s": max_gap_by_day,
            "rates_hz": {d: s.to_dict() for d, s in rates_by_day.items()},
            "pair": pair_for_0e,
        }
    if stop_0e is not None:
        verdicts["0e"] = f"ACTIVE ELECTRODES STOPPED: {stop_0e}"
        report += [f"STOPPED: {stop_0e}", ""]
        sidecar["task_0e"] = {"stopped": stop_0e}

    sidecar["verdicts"] = verdicts
    report += ["## Verdicts", ""]
    report += [verdicts[k] for k in ("0a", "0b", "0c", "0d", "0e")]
    with open(os.path.join(out_dir, "step0_report.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(out_dir, "step0.json"), "w") as f:
        json.dump(sidecar, f, indent=2, default=_json_default)
    for k in ("0a", "0b", "0c", "0d", "0e"):
        logger.info("%s: %s", k, verdicts[k])
    return sidecar


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Gate 1 Step 0: resolve the open questions.")
    parser.add_argument("--config", type=str, default="configs/recovery_gate1.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    run_step0(args.config)
