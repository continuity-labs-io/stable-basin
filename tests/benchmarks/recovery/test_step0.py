"""Unit tests for Gate 1 Step 0 (src/benchmarks/recovery/step0.py).

All HDF5 fixtures are synthetic, written to tmp_path with pandas/PyTables in the
same table layout as fs369_package.hdf5 (table format, ``time_of_*`` and
``electrode`` as data columns, one-row fixed-format metadata).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.benchmarks.recovery import step0
from src.benchmarks.recovery.step0 import (
    adjacent_pair_verdict,
    build_bin_edges_ns,
    check_input_path,
    choose_window,
    classify_locality,
    collect_peristimulus,
    compute_gaps,
    coverage_table,
    day_pairs,
    day_table,
    exclusion_intervals,
    filter_candidate,
    propose_blanking,
    psth_table,
    quiet_rates,
    run_step0,
    slot_lookup,
)

NS = 1_000_000_000
MS = 1_000_000

CANDIDATE = {
    "a1": 0.8,
    "a2": 0.8,
    "d1": 300,
    "d2": 300,
    "nb_stim_pulse": 2,
    "pulse_train_period": 10000,
    "stim_polarity": 0,
    "stim_shape": 0,
}
SLOTS = [[64, 71], [72, 79], [80, 87], [88, 95]]
POST_SEGMENTS = [[0.0, 20.0, 0.1], [20.0, 100.0, 1.0], [100.0, 500.0, 10.0]]
TABLES = {
    "stimulations": "/fs369_wholelife_stimulations",
    "events": "/fs369_wholelife_events",
    "metadata": "/fs369_wholelife_metadata",
    "door": "/fs369_wholelife_incubator_door_opening",
}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _stim_rows(times: list[pd.Timestamp], electrodes: list[int], **overrides) -> pd.DataFrame:
    params = {**CANDIDATE, **overrides}
    df = pd.DataFrame({"electrode": np.asarray(electrodes, dtype=np.int64), "time_of_stim": pd.to_datetime(times)})
    for k, v in params.items():
        df[k] = float(v)
    return df


def _write_package(
    path: Path,
    stim: pd.DataFrame,
    events: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    door: pd.DataFrame | None = None,
) -> None:
    if metadata is None:
        metadata = pd.DataFrame({"Date": ["2025-02-19"], "medium changes": ["['2025-02-28T14:01:34']"]})
    if door is None:
        door = pd.DataFrame(
            {
                "time": pd.to_datetime(["2025-02-19 08:00:00", "2025-02-19 08:01:00"]),
                "incubator_id": ["0", "0"],
                "incubator_door_opening": [True, False],
            }
        )
    with pd.HDFStore(path, "w") as st:
        st.put(TABLES["stimulations"], stim.reset_index(drop=True), format="table",
               data_columns=["time_of_stim", "electrode"])
        st.put(TABLES["events"], events.reset_index(drop=True), format="table",
               data_columns=["time_of_event", "electrode"])
        st.put(TABLES["door"], door, format="table", data_columns=["time"])
        st.put(TABLES["metadata"], metadata, format="fixed")


def _write_config(tmp_path: Path, input_path: Path, expected: dict) -> Path:
    cfg = {
        "input_path": str(input_path),
        "forbidden_input_names": ["fs369_raw.hdf5"],
        "output_dir": str(tmp_path / "out"),
        "tables": TABLES,
        "metadata_fields": {"first_recording_date": "Date", "medium_changes": "medium changes"},
        "candidate": CANDIDATE,
        "expected_candidate": expected,
        "slots": SLOTS,
        "event_query_margin_s": 30.0,
        "psth": {
            "pre_window_ms": 50.0,
            "post_window_ms": 500.0,
            "post_segments": POST_SEGMENTS,
            "voltage_percentile": 95.0,
            "report_summary_edges_ms": [-50, 0, 5, 20, 100, 500],
        },
        "blanking": {"threshold_multiplier": 3.0, "n_consecutive_bins": 5, "local_ratio": 2.0},
        "windows": {
            "gap_percentiles": [0, 1, 5, 10, 25, 50, 75, 100],
            "candidate_lengths_s": [0.5, 1, 2, 5, 10, 20, 30, 60],
            "min_fraction_stims": 0.6,
            "min_days_per_electrode": 2,
        },
        "activity": {"rate_thresholds_hz": [0.1, 0.5, 1.0], "verdict_threshold_hz": 0.1},
        "seed": 0,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _three_day_package(tmp_path: Path, metadata: pd.DataFrame | None = None) -> tuple[Path, int]:
    """CANDIDATE stims on electrodes 64 and 72, 10 triggers 10 s apart, on 3 days."""
    rng = np.random.default_rng(0)
    days = [pd.Timestamp("2025-02-20 12:00:00"), pd.Timestamp("2025-02-21 12:00:00"), pd.Timestamp("2025-02-28 16:00:00")]
    stim_parts, ev_parts = [], []
    for d in days:
        times = [d + pd.Timedelta(seconds=10 * i) for i in range(10)]
        stim_parts.append(_stim_rows(times * 2, [64] * 10 + [72] * 10))
        start, stop = d - pd.Timedelta(seconds=60), d + pd.Timedelta(seconds=150)
        for e in (64, 72, 80):
            n = rng.poisson(20.0 * (stop - start).total_seconds())
            t = np.sort(rng.integers(start.value, stop.value, size=n))
            ev_parts.append(
                pd.DataFrame({"electrode": np.full(n, e, np.int64), "time_of_event": pd.to_datetime(t),
                              "max_voltage_uv": rng.normal(-50, 5, n)})
            )
    stim = pd.concat(stim_parts).sort_values("time_of_stim", kind="stable")
    events = pd.concat(ev_parts).sort_values("time_of_event", kind="stable")
    path = tmp_path / "fs_test_package.hdf5"
    _write_package(path, stim, events, metadata=metadata)
    return path, len(days)


# ---------------------------------------------------------------------------
# Data access (written first: it is the one that matters for runtime)
# ---------------------------------------------------------------------------


def test_events_never_loaded_whole(tmp_path, monkeypatch):
    # ARRANGE
    pkg, n_days = _three_day_package(tmp_path)
    cfg = _write_config(tmp_path, pkg, {"n_stims": 60, "n_electrodes": 2, "n_days": 3})
    calls: list[tuple[str, object]] = []
    original = pd.HDFStore.select

    def spy(self, key, *args, **kwargs):
        calls.append((key, kwargs.get("where", args[0] if args else None)))
        return original(self, key, *args, **kwargs)

    monkeypatch.setattr(pd.HDFStore, "select", spy)

    # ACT
    sidecar = run_step0(str(cfg))

    # ASSERT
    assert len(calls) == n_days
    assert all(where is not None and "time_of_event" in str(where) for _, where in calls)
    assert all(key.lstrip("/") == TABLES["events"].lstrip("/") for key, _ in calls)
    assert sidecar["events_query"]["n_selects"] == n_days
    assert set(sidecar["events_query"]["rows_per_day"]) == {"2025-02-20", "2025-02-21", "2025-02-28"}


def test_report_ends_with_five_verdicts_and_writes_outputs(tmp_path):
    # ARRANGE
    pkg, _ = _three_day_package(tmp_path)
    cfg = _write_config(tmp_path, pkg, {"n_stims": 60, "n_electrodes": 2, "n_days": 3})
    out = tmp_path / "out"
    prefixes = ["ADJACENT PAIR", "BLANKING", "RESPONSE WINDOW", "BASELINE WINDOW", "ACTIVE ELECTRODES"]
    names = ["step0_report.md", "step0.json", "stim_windows.csv", "artifact_psth.csv", "artifact_psth.png"]

    # ACT
    sidecar = run_step0(str(cfg))
    lines = (out / "step0_report.md").read_text().rstrip("\n").split("\n")
    windows = pd.read_csv(out / "stim_windows.csv")

    # ASSERT
    assert all((out / n).is_file() for n in names)
    for line, prefix in zip(lines[-5:], prefixes):
        assert line.startswith(prefix)
    assert lines[-6] == ""
    assert lines[-5] == "ADJACENT PAIR: 2025-02-20, 2025-02-21"
    assert lines[-3] == "RESPONSE WINDOW 10 s"
    assert list(windows.columns) == [
        "stim_id", "electrode", "time_of_stim", "day", "gap_before_same", "gap_before_any",
        "gap_after_same", "gap_after_any", "usable",
    ]
    assert len(windows) == 60
    assert sidecar["slot_assumption"]["status"] == "UNVERIFIED"
    assert sidecar["seed"] == 0
    assert "git_commit" in sidecar["code_version"]


def test_missing_medium_changes_stops_task_0a_and_0e(tmp_path):
    # ARRANGE
    md = pd.DataFrame({"Date": ["2025-02-19"]})
    pkg, _ = _three_day_package(tmp_path, metadata=md)
    cfg = _write_config(tmp_path, pkg, {"n_stims": 60, "n_electrodes": 2, "n_days": 3})

    # ACT
    sidecar = run_step0(str(cfg))

    # ASSERT
    assert sidecar["verdicts"]["0a"].startswith("ADJACENT PAIR STOPPED")
    assert "medium changes" in sidecar["verdicts"]["0a"]
    assert sidecar["verdicts"]["0e"].startswith("ACTIVE ELECTRODES STOPPED")
    assert sidecar["verdicts"]["0c"] == "RESPONSE WINDOW 10 s"


def test_check_input_path_refuses_raw_file(tmp_path):
    # ARRANGE
    raw = tmp_path / "fs369_raw.hdf5"
    raw.write_bytes(b"")
    missing = tmp_path / "absent_package.hdf5"

    # ACT / ASSERT
    with pytest.raises(ValueError, match="forbidden"):
        check_input_path(str(raw), ["fs369_raw.hdf5"])
    with pytest.raises(FileNotFoundError):
        check_input_path(str(missing), ["fs369_raw.hdf5"])


# ---------------------------------------------------------------------------
# Task 0a
# ---------------------------------------------------------------------------


def test_candidate_filter_exact():
    # ARRANGE
    t = [pd.Timestamp("2025-02-20 12:00:00")]
    exact = _stim_rows(t * 2, [64, 65])
    near_misses = [
        _stim_rows(t, [70], a1=0.9),
        _stim_rows(t, [70], a2=0.7),
        _stim_rows(t, [70], d1=200),
        _stim_rows(t, [70], d2=301),
        _stim_rows(t, [70], nb_stim_pulse=1),
        _stim_rows(t, [70], pulse_train_period=20000),
        _stim_rows(t, [70], stim_polarity=1),
        _stim_rows(t, [70], stim_shape=1),
    ]
    stim = pd.concat([exact, *near_misses], ignore_index=True)

    # ACT
    got = filter_candidate(stim, CANDIDATE)

    # ASSERT
    assert len(got) == 2
    assert sorted(got["electrode"].tolist()) == [64, 65]
    assert sorted(got.index.tolist()) == [0, 1]
    with pytest.raises(KeyError):
        filter_candidate(stim.drop(columns=["d2"]), CANDIDATE)


def _days_df(day_numbers: list[int]) -> pd.DataFrame:
    times, electrodes = [], []
    for d in day_numbers:
        base = pd.Timestamp(f"2025-02-{d:02d} 12:00:00")
        times += [base, base + pd.Timedelta(seconds=10)]
        electrodes += [64, 72]
    cand = filter_candidate(_stim_rows(times, electrodes), CANDIDATE)
    return day_table(cand, pd.Timestamp("2025-01-31"))


def test_adjacent_pair_detection():
    # ARRANGE
    days_a = _days_df([1, 2, 9])
    days_b = _days_df([1, 5, 9])

    # ACT
    verdict_a, pair_a = adjacent_pair_verdict(day_pairs(days_a, []))
    verdict_b, pair_b = adjacent_pair_verdict(day_pairs(days_b, []))

    # ASSERT
    assert verdict_a == "ADJACENT PAIR: 2025-02-01, 2025-02-02"
    assert pair_a == ("2025-02-01", "2025-02-02")
    assert verdict_b.startswith("NO ADJACENT PAIR")
    assert "2025-02-01, 2025-02-05" in verdict_b and "(4 days apart)" in verdict_b
    assert days_a["days_since_first_recording"].tolist() == [1, 2, 9]


def test_medium_change_disqualifies_pair():
    # ARRANGE
    days = _days_df([1, 2, 9])
    change = [pd.Timestamp("2025-02-01 20:00:00")]

    # ACT
    pairs = day_pairs(days, change)
    verdict, _ = adjacent_pair_verdict(pairs)

    # ASSERT
    first = pairs[(pairs["day_a"] == "2025-02-01") & (pairs["day_b"] == "2025-02-02")].iloc[0]
    assert bool(first["disqualified"]) is True
    assert verdict.startswith("NO ADJACENT PAIR")
    assert "2025-02-02, 2025-02-09" in verdict


# ---------------------------------------------------------------------------
# Task 0b
# ---------------------------------------------------------------------------


def _synthetic_psth(with_artifact: bool) -> pd.DataFrame:
    """2000 stims 1 s apart on electrode 64; 200 Hz Poisson baseline; optional 4 ms burst."""
    rng = np.random.default_rng(1)
    n_stims, rate_hz = 2000, 200.0
    stim_t = (np.arange(n_stims, dtype=np.int64) + 1) * NS
    span = (0, (n_stims + 1) * NS)
    n_base = rng.poisson(rate_hz * (span[1] - span[0]) / NS)
    ev_t = [rng.integers(span[0], span[1], size=n_base)]
    ev_v = [np.full(n_base, -50.0)]
    if with_artifact:
        per_stim = 40
        burst = (stim_t[:, None] + rng.integers(0, 4 * MS, size=(n_stims, per_stim))).ravel()
        ev_t.append(burst)
        ev_v.append(np.full(burst.size, 5000.0))
    t = np.concatenate(ev_t)
    v = np.concatenate(ev_v)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    e = np.full(t.size, 64, dtype=np.int64)
    slot_of = slot_lookup(SLOTS)
    edges = build_bin_edges_ns(50.0, 500.0, POST_SEGMENTS)
    got = collect_peristimulus(stim_t, np.full(n_stims, 64), t, e, v, 50 * MS, 500 * MS, slot_of)
    denom = np.array([n_stims, n_stims * 7, n_stims * 24])
    return psth_table(got["rel_ns"], got["category"], got["voltage"], edges, denom, 95.0)


def test_blanking_recovers_known_artifact():
    # ARRANGE
    psth = _synthetic_psth(with_artifact=True)

    # ACT
    prop = propose_blanking(
        "stim_electrode", psth["bin_start_ms"].to_numpy(), psth["bin_end_ms"].to_numpy(),
        psth["rate_hz_stim_electrode"].to_numpy(), 3.0, 5,
    )

    # ASSERT
    assert prop.blanking_ms is not None
    assert abs(prop.blanking_ms - 4.0) <= 1.0
    assert prop.pre_stimulus_median_rate_hz == pytest.approx(200.0, rel=0.2)
    assert psth.loc[psth["bin_start_ms"] == 1.0, "voltage_median_uv_stim_electrode"].iloc[0] == 5000.0


def test_blanking_returns_nothing_on_clean_data():
    # ARRANGE
    psth = _synthetic_psth(with_artifact=False)
    first_post_bin = float(psth.loc[psth["bin_start_ms"] >= 0, "bin_start_ms"].iloc[0])

    # ACT
    prop = propose_blanking(
        "stim_electrode", psth["bin_start_ms"].to_numpy(), psth["bin_end_ms"].to_numpy(),
        psth["rate_hz_stim_electrode"].to_numpy(), 3.0, 5,
    )

    # ASSERT
    assert prop.blanking_ms == first_post_bin == 0.0
    assert prop.reason is None


def test_blanking_zero_pre_median_is_undetermined():
    # ARRANGE
    start = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    end = start + 0.5
    rate = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ACT
    prop = propose_blanking("stim_electrode", start, end, rate, 3.0, 5)

    # ASSERT
    assert prop.blanking_ms is None
    assert "median rate is 0" in prop.reason
    assert prop.longest_zero_run_ms == (0.5, 3.0)


def test_bin_edges_mirror_and_reject_gaps():
    # ARRANGE
    expected_ms = {-50.0, -20.0, -0.1, 0.0, 0.1, 20.0, 21.0, 100.0, 110.0, 500.0}

    # ACT
    edges_ms = build_bin_edges_ns(50.0, 500.0, POST_SEGMENTS) / MS

    # ASSERT
    assert expected_ms.issubset(set(np.round(edges_ms, 6).tolist()))
    assert edges_ms[0] == -50.0 and edges_ms[-1] == 500.0
    assert np.all(np.diff(edges_ms) > 0)
    assert np.isclose(np.diff(edges_ms[(edges_ms >= -20) & (edges_ms <= 0)]), 0.1).all()
    with pytest.raises(ValueError):
        build_bin_edges_ns(50.0, 500.0, [[0.0, 20.0, 0.1], [30.0, 500.0, 10.0]])
    with pytest.raises(ValueError):
        build_bin_edges_ns(50.0, 500.0, [[0.0, 20.0, 0.3], [20.0, 500.0, 10.0]])


def test_collect_peristimulus_categories_and_unmapped():
    # ARRANGE
    slot_of = slot_lookup(SLOTS)
    stim_t = np.array([1 * NS], dtype=np.int64)
    ev_t = np.array([1 * NS + 1 * MS, 1 * NS + 2 * MS, 1 * NS + 3 * MS, 1 * NS + 4 * MS, 3 * NS], dtype=np.int64)
    ev_e = np.array([64, 65, 90, 200, 64])
    ev_v = np.zeros(5)

    # ACT
    got = collect_peristimulus(stim_t, np.array([64]), ev_t, ev_e, ev_v, 50 * MS, 500 * MS, slot_of)

    # ASSERT
    assert got["category"].tolist() == [0, 1, 2]
    assert got["rel_ns"].tolist() == [1 * MS, 2 * MS, 3 * MS]
    assert got["n_unmapped"] == 1
    with pytest.raises(ValueError):
        collect_peristimulus(stim_t, np.array([10]), ev_t, ev_e, ev_v, 50 * MS, 500 * MS, slot_of)


def test_classify_locality():
    # ARRANGE / ACT / ASSERT
    assert classify_locality(4.0, 4.5, 0.1, 2.0) == "GLOBAL"
    assert classify_locality(10.0, 4.0, 0.1, 2.0) == "LOCAL"
    assert classify_locality(0.0, 0.1, 0.1, 2.0) == "GLOBAL"
    assert classify_locality(None, 4.0, 0.1, 2.0) is None


# ---------------------------------------------------------------------------
# Tasks 0c / 0d
# ---------------------------------------------------------------------------


def test_gap_computation_same_vs_any_electrode():
    # ARRANGE
    all_t = np.array([0, 5, 10, 15, 20, 20], dtype=np.int64) * NS
    all_e = np.array([64, 72, 64, 72, 64, 72])

    # ACT
    g = compute_gaps(all_t, all_e, all_t, all_e)

    # ASSERT
    i = 2  # electrode 64 at t = 10 s
    assert g["gap_after_same"][i] == 10.0 and g["gap_after_any"][i] == 5.0
    assert g["gap_before_same"][i] == 10.0 and g["gap_before_any"][i] == 5.0
    j = 4  # electrode 64 at t = 20 s, co-timed with electrode 72
    assert g["gap_before_any"][j] == 5.0
    assert np.isinf(g["gap_after_any"][j]) and np.isinf(g["gap_after_same"][j])
    assert np.isinf(g["gap_before_same"][0])


def test_choose_window_requires_electrodes_on_min_days():
    # ARRANGE
    gaps = np.array([12.0, 12.0, 12.0, 3.0])
    electrodes = np.array([64, 64, 72, 72])
    days = np.array(["d1", "d2", "d1", "d2"])

    # ACT
    cov = coverage_table(gaps, gaps, electrodes, days, [1.0, 10.0], min_days=2)
    w_strict = choose_window(cov, n_required_electrodes=2, min_fraction=0.6)
    w_loose = choose_window(cov, n_required_electrodes=1, min_fraction=0.6)

    # ASSERT
    assert cov["n_stims"].tolist() == [4, 3]
    assert cov["n_electrodes_on_min_days"].tolist() == [2, 1]
    assert w_strict == 1.0
    assert w_loose == 10.0


# ---------------------------------------------------------------------------
# Task 0e
# ---------------------------------------------------------------------------


def test_quiet_rates_exclude_stim_spans():
    # ARRANGE
    span = (0, 100 * NS)
    stim_t = np.array([40 * NS, 45 * NS], dtype=np.int64)
    excl = exclusion_intervals(stim_t, 10 * NS, 10 * NS, span)
    ev_t = np.array([5, 35, 50, 60, 95], dtype=np.int64) * NS
    ev_e = np.array([64, 64, 64, 72, 72])

    # ACT
    rates, quiet_s = quiet_rates(ev_t, ev_e, [64, 72, 80], span, excl)

    # ASSERT
    assert excl.tolist() == [[30 * NS, 55 * NS]]
    assert quiet_s == 75.0
    assert rates.loc[64] == pytest.approx(1 / 75.0)
    assert rates.loc[72] == pytest.approx(2 / 75.0)
    assert rates.loc[80] == 0.0
    with pytest.raises(step0.TaskStopped):
        quiet_rates(ev_t, ev_e, [64], span, np.array([[0, 100 * NS]], dtype=np.int64))
