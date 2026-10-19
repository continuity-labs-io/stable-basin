"""Unit tests for Stage A inventory and Stage B probe (src/benchmarks/recovery/probe.py).

All HDF5 fixtures are synthetic and built in tmp_path (a few channels, a few
hundred frames). Sample rates are chosen low so that tiny frame counts still give
durations on both sides of the skip + window threshold.
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from src.benchmarks.recovery.probe import (
    FileEntry,
    infer_group_token,
    inventory,
    probe,
    run_census_probe,
    segment_sessions,
)

SKIP_S = 60.0
WINDOW_S = 40.0
GAP_S = 600.0

ALL_PATTERNS = {
    "finalspark": ["**/*_package.hdf5"],
    "maxwell": ["**/*.raw.h5"],
    "organoid_pharma": ["**/*.raw.h5"],
    "threebrain": ["**/*.brw"],
}


def _entry(path: Path, source: str) -> FileEntry:
    return FileEntry(str(path), source, path.stat().st_size, path.stat().st_mtime)


def _probe_one(entry: FileEntry):
    probes, assumptions = probe(
        entry,
        census_skip_s=SKIP_S,
        census_window_len_s=WINDOW_S,
        finalspark_session_gap_s=GAP_S,
    )
    assert len(probes) == 1
    return probes[0], assumptions


def _write_maxwell_v1(
    path: Path,
    n_channels: int,
    n_frames: int,
    sampling_rate: float | None,
    with_mapping: bool = True,
) -> None:
    with h5py.File(path, "w") as f:
        sig = f.create_dataset("/sig", data=np.zeros((n_channels, n_frames), dtype="uint16"))
        if sampling_rate is not None:
            sig.attrs["sampling_rate"] = sampling_rate
        f.create_dataset("time", data=np.array([b"start: 2026-01-01 10:00:00;\nstop: x\n"]))
        f.create_dataset("version", data=np.array([b"20160704"]))
        f.create_dataset("settings/gain", data=np.array([512.0]))
        if with_mapping:
            mapping_dtype = np.dtype(
                [("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")]
            )
            mapping = np.zeros(n_channels, dtype=mapping_dtype)
            mapping["x"] = np.arange(n_channels) * 17.5
            f.create_dataset("/mapping", data=mapping)


def _write_threebrain(
    path: Path,
    n_rows: int | None,
    n_cols: int | None,
    n_frames: int,
    sampling_rate: float | None,
    pitch_um: float | None,
) -> None:
    n_ch = (n_rows or 2) * (n_cols or 2)
    with h5py.File(path, "w") as f:
        f.create_dataset("3BData/Raw", data=np.zeros(n_ch * n_frames, dtype="uint16"))
        if n_rows is not None and n_cols is not None:
            f.create_dataset("3BRecInfo/3BMeaChip/NRows", data=np.array([n_rows]))
            f.create_dataset("3BRecInfo/3BMeaChip/NCols", data=np.array([n_cols]))
        if pitch_um is not None:
            f.create_dataset("3BRecInfo/3BMeaChip/Pitch", data=np.array([pitch_um]))
        if sampling_rate is not None:
            f.create_dataset("3BRecInfo/3BRecVars/SamplingRate", data=np.array([sampling_rate]))
        f.create_dataset("3BUserInfo/ExpDate", data="2026-05-01")


def _write_finalspark_events(path: Path, times: list[str], electrodes: list[int]) -> None:
    events = pd.DataFrame(
        {
            "electrode": np.asarray(electrodes, dtype=np.int64),
            "time_of_event": pd.to_datetime(times),
            "max_voltage_uv": np.linspace(-60.0, -30.0, len(times)),
        }
    )
    events.to_hdf(
        path,
        key="fs437_wholelife_events",
        mode="w",
        format="table",
        data_columns=["time_of_event", "electrode"],
    )


# ---------------------------------------------------------------------------
# infer_group_token
# ---------------------------------------------------------------------------


def test_infer_group_token():
    # ARRANGE
    cases = [
        ("data/ephys/pharmacological_shock/Drug_2953_control.raw.h5", "Drug_2953"),
        ("data/ephys/pharmacological_shock/Drug_2953_30uM.raw.h5", "Drug_2953"),
        ("data/ephys/finalspark/fs437_export/fs437_package.hdf5", "fs437"),
        ("data/ephys/hdmea_neuropulse.brw", "hdmea_neuropulse"),
        ("data/ephys/sample_recording.raw.h5", "sample_recording"),
    ]

    for path, expected in cases:
        # ACT
        actual = infer_group_token(path)

        # ASSERT
        assert actual == expected


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_resolves_patterns_under_each_root(tmp_path: Path):
    # ARRANGE
    fs_dir = tmp_path / "finalspark" / "fs437_export"
    mx_dir = tmp_path / "maxwell"
    tb_dir = tmp_path / "threebrain"
    op_dir = tmp_path / "organoid_pharma"
    for d in [fs_dir, mx_dir, tb_dir, op_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (fs_dir / "fs437_package.hdf5").write_bytes(b"package")
    (mx_dir / "chip1.raw.h5").write_bytes(b"raw_h5")
    (tb_dir / "neuropulse.brw").write_bytes(b"brw")
    (op_dir / "Drug_1_control.raw.h5").write_bytes(b"op_raw_h5")

    (fs_dir / "fs437_raw.hdf5").write_bytes(b"ignore")
    (mx_dir / "plain.h5").write_bytes(b"ignore")
    (mx_dir / "notes.txt").write_bytes(b"ignore")
    (op_dir / ".hidden.raw.h5").write_bytes(b"ignore")

    roots = {
        "finalspark": str(tmp_path / "finalspark"),
        "maxwell": str(mx_dir),
        "threebrain": str(tb_dir),
        "organoid_pharma": str(op_dir),
        "missing_source": str(tmp_path / "nonexistent"),
    }
    patterns = {**ALL_PATTERNS, "missing_source": ["**/*.h5"]}

    # ACT
    entries = inventory(roots, patterns)

    # ASSERT
    assert [e.source for e in entries] == ["finalspark", "maxwell", "organoid_pharma", "threebrain"]
    names = [Path(e.path).name for e in entries]
    assert names == ["fs437_package.hdf5", "chip1.raw.h5", "Drug_1_control.raw.h5", "neuropulse.brw"]
    assert all(e.bytes > 0 for e in entries)


def test_inventory_raises_when_two_roots_match_same_file(tmp_path: Path):
    # ARRANGE
    parent = tmp_path / "ephys"
    child = parent / "pharmacological_shock"
    child.mkdir(parents=True)
    (child / "Drug_1_control.raw.h5").write_bytes(b"x")
    roots = {"maxwell": str(parent), "organoid_pharma": str(child)}
    patterns = {"maxwell": ["**/*.raw.h5"], "organoid_pharma": ["**/*.raw.h5"]}

    # ACT / ASSERT
    with pytest.raises(ValueError, match="resolved by both sources"):
        inventory(roots, patterns)


def test_inventory_raises_when_source_has_no_patterns(tmp_path: Path):
    # ARRANGE
    roots = {"maxwell": str(tmp_path), "threebrain": str(tmp_path)}
    patterns = {"maxwell": ["**/*.raw.h5"]}

    # ACT / ASSERT
    with pytest.raises(ValueError, match="No glob patterns configured"):
        inventory(roots, patterns)


def test_inventory_dedupes_overlapping_patterns_within_one_source(tmp_path: Path):
    # ARRANGE
    (tmp_path / "a.raw.h5").write_bytes(b"x")
    roots = {"maxwell": str(tmp_path)}
    patterns = {"maxwell": ["**/*.raw.h5", "*.raw.h5"]}

    # ACT
    entries = inventory(roots, patterns)

    # ASSERT
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# MaxWell
# ---------------------------------------------------------------------------


def test_probe_maxwell_v1_with_sampling_rate(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "rec.raw.h5"
    n_channels, n_frames, rate = 4, 300, 2.5  # 120 s
    _write_maxwell_v1(path, n_channels, n_frames, rate)

    # ACT
    p, a = _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert p.probe_error is None
    assert p.sample_rate_hz == pytest.approx(rate)
    assert p.duration_s == pytest.approx(n_frames / rate)
    assert p.n_channels == n_channels
    assert p.maxwell_layout == "v1"
    assert p.has_xy is True
    assert p.xy_units == "um"
    assert p.xy_provenance == "mapping_dataset"
    assert p.recording_timestamp == "2026-01-01 10:00:00"
    assert p.session_index is None
    assert a["sample_rate_source"] == "sig.attrs['sampling_rate']"
    assert a["signal_key"] == "/sig"


def test_probe_maxwell_without_sampling_rate_sets_error(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "norate.raw.h5"
    _write_maxwell_v1(path, 4, 300, sampling_rate=None)

    # ACT
    p, a = _probe_one(_entry(path, "organoid_pharma"))

    # ASSERT
    assert p.sample_rate_hz is None
    assert p.duration_s is None
    assert p.probe_error is not None
    assert p.probe_error.startswith("sample_rate not found in")
    assert "/data_store/data0000/settings/sampling" in p.probe_error
    assert p.n_channels == 4
    assert p.maxwell_layout == "v1"
    assert a["sample_rate_source"] is None
    assert "sig" in a["top_level_keys"]
    assert "settings/gain" in a["key_tree"]
    assert a["version"] == "20160704"


@pytest.mark.parametrize(
    "location,expected_source",
    [
        ("attr:sampling", "sig.attrs['sampling']"),
        ("attr:samplingRate", "sig.attrs['samplingRate']"),
        ("attr:fs", "sig.attrs['fs']"),
        ("/settings/sampling", "/settings/sampling"),
        ("/settings/sampling_rate", "/settings/sampling_rate"),
        ("/sampling_rate", "/sampling_rate"),
    ],
)
def test_probe_maxwell_v1_sample_rate_lookup_chain(
    tmp_path: Path, location: str, expected_source: str
):
    # ARRANGE
    path = tmp_path / "chain.raw.h5"
    rate = 2.0
    with h5py.File(path, "w") as f:
        sig = f.create_dataset("/sig", data=np.zeros((3, 300), dtype="uint16"))
        if location.startswith("attr:"):
            sig.attrs[location.split(":", 1)[1]] = rate
        else:
            f.create_dataset(location, data=np.array([rate]))

    # ACT
    p, a = _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert a["sample_rate_source"] == expected_source
    assert p.sample_rate_hz == pytest.approx(rate)
    assert p.duration_s == pytest.approx(150.0)
    assert p.probe_error is None


def test_probe_maxwell_lookup_prefers_earlier_entry(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "both.raw.h5"
    with h5py.File(path, "w") as f:
        sig = f.create_dataset("/sig", data=np.zeros((2, 300), dtype="uint16"))
        sig.attrs["fs"] = 2.0
        f.create_dataset("/settings/sampling", data=np.array([999.0]))

    # ACT
    p, a = _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert a["sample_rate_source"] == "sig.attrs['fs']"
    assert p.sample_rate_hz == pytest.approx(2.0)


@pytest.mark.parametrize(
    "rate_path",
    [
        "/recordings/rec0000/well000/settings/sampling",
        "/data_store/data0000/settings/sampling",
    ],
)
def test_probe_maxwell_v2_layout_reports_rate_and_missing_signal(tmp_path: Path, rate_path: str):
    # ARRANGE
    path = tmp_path / "v2.raw.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(rate_path, data=np.array([20000.0]))

    # ACT
    p, a = _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert p.maxwell_layout == "v2"
    assert a["sample_rate_source"] == rate_path
    assert p.sample_rate_hz == pytest.approx(20000.0)
    assert p.n_channels is None
    assert p.duration_s is None
    assert p.probe_error is not None and "signal dataset not found" in p.probe_error


def test_probe_maxwell_routing_signal_key(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "routing.raw.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("/routing/lsb", data=np.zeros((8, 300), dtype="uint16"))
        f.create_dataset("/settings/sampling", data=np.array([2.0]))

    # ACT
    p, a = _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert a["signal_key"] == "/routing/lsb"
    assert p.n_channels == 8
    assert p.maxwell_layout == "unknown"
    assert p.xy_units == "absent"
    assert p.has_xy is False


# ---------------------------------------------------------------------------
# Duration threshold (skip + window)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_frames,expect_error",
    [
        (150, True),  # 60 s: <= skip alone
        (200, True),  # 80 s: > skip but <= skip + window
        (250, True),  # 100 s: == skip + window (boundary)
        (300, False),  # 120 s: > skip + window
    ],
)
def test_probe_duration_threshold_uses_skip_plus_window(
    tmp_path: Path, n_frames: int, expect_error: bool
):
    # ARRANGE
    path = tmp_path / "dur.raw.h5"
    _write_maxwell_v1(path, 2, n_frames, sampling_rate=2.5)

    # ACT
    p, _ = _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert p.duration_s == pytest.approx(n_frames / 2.5)
    if expect_error:
        assert p.probe_error is not None
        assert "<= census_skip_s + census_window_len_s (100.0s)" in p.probe_error
    else:
        assert p.probe_error is None


def test_probe_warns_on_probe_error(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # ARRANGE
    path = tmp_path / "short.raw.h5"
    _write_maxwell_v1(path, 2, 100, sampling_rate=2.5)

    # ACT
    with caplog.at_level("WARNING", logger="src.benchmarks.recovery.probe"):
        _probe_one(_entry(path, "maxwell"))

    # ASSERT
    assert any(r.levelname == "WARNING" for r in caplog.records)


# ---------------------------------------------------------------------------
# 3Brain
# ---------------------------------------------------------------------------


def test_probe_threebrain_with_pitch_is_um(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "pitch.brw"
    _write_threebrain(path, n_rows=2, n_cols=3, n_frames=300, sampling_rate=2.0, pitch_um=42.0)

    # ACT
    p, a = _probe_one(_entry(path, "threebrain"))

    # ASSERT
    assert p.has_xy is True
    assert p.xy_units == "um"
    assert p.xy_provenance == "grid_times_pitch"
    assert p.n_channels == 6
    assert p.n_rows == 2 and p.n_cols == 3
    assert p.sample_rate_hz == pytest.approx(2.0)
    assert p.duration_s == pytest.approx(150.0)
    assert p.probe_error is None
    assert a["pitch_um"] == pytest.approx(42.0)
    assert p.maxwell_layout is None


def test_probe_threebrain_without_pitch_is_grid_index(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "nopitch.brw"
    _write_threebrain(path, n_rows=2, n_cols=2, n_frames=300, sampling_rate=2.0, pitch_um=None)

    # ACT
    p, a = _probe_one(_entry(path, "threebrain"))

    # ASSERT
    assert p.has_xy is True
    assert p.xy_units == "grid_index"
    assert p.xy_provenance == "derived_from_grid"
    assert a["pitch"] == "not_found"
    assert p.probe_error is None


def test_probe_threebrain_missing_grid_and_rate_set_errors(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "bare.brw"
    _write_threebrain(path, n_rows=None, n_cols=None, n_frames=300, sampling_rate=None, pitch_um=None)

    # ACT
    p, a = _probe_one(_entry(path, "threebrain"))

    # ASSERT
    assert p.n_rows is None and p.n_cols is None and p.n_channels is None
    assert p.sample_rate_hz is None
    assert p.duration_s is None
    assert p.has_xy is False
    assert p.xy_units == "absent"
    assert "grid dimensions not found" in p.probe_error
    assert "sample_rate not found in ['3BRecInfo/3BRecVars/SamplingRate']" in p.probe_error
    assert a["grid_source"] is None


def test_probe_threebrain_short_recording_flagged(tmp_path: Path):
    # ARRANGE: 89.9 s, like the real 3Brain file, is below skip + window (100 s).
    path = tmp_path / "short.brw"
    _write_threebrain(path, n_rows=2, n_cols=2, n_frames=899, sampling_rate=10.0, pitch_um=None)

    # ACT
    p, _ = _probe_one(_entry(path, "threebrain"))

    # ASSERT
    assert p.duration_s == pytest.approx(89.9)
    assert p.probe_error is not None and "census_window_len_s" in p.probe_error


# ---------------------------------------------------------------------------
# segment_sessions
# ---------------------------------------------------------------------------


def test_segment_sessions_boundaries():
    # ARRANGE
    s = 1_000_000_000
    times = np.array([0, 10 * s, 610 * s, 1211 * s, 1212 * s], dtype=np.int64)
    gap_s = 600.0

    # ACT
    sessions = segment_sessions(times, gap_s)

    # ASSERT: a 600 s gap does not split; a 601 s gap does.
    assert sessions == [(0, 3), (3, 5)]
    assert sum(b - a for a, b in sessions) == times.size


def test_segment_sessions_empty_and_single():
    # ARRANGE
    empty = np.array([], dtype=np.int64)
    single = np.array([5], dtype=np.int64)

    # ACT
    out_empty = segment_sessions(empty, 600.0)
    out_single = segment_sessions(single, 600.0)

    # ASSERT
    assert out_empty == []
    assert out_single == [(0, 1)]


@pytest.mark.parametrize(
    "times,gap_s,match",
    [
        (np.array([3, 1, 2], dtype=np.int64), 600.0, "sorted"),
        (np.array([1, 2], dtype=np.int64), 0.0, "positive"),
        (np.zeros((2, 2), dtype=np.int64), 600.0, "1-D"),
    ],
)
def test_segment_sessions_rejects_bad_input(times: np.ndarray, gap_s: float, match: str):
    # ARRANGE (parametrized)

    # ACT / ASSERT
    with pytest.raises(ValueError, match=match):
        segment_sessions(times, gap_s)


# ---------------------------------------------------------------------------
# FinalSpark
# ---------------------------------------------------------------------------


def test_probe_finalspark_two_sessions_two_hour_gap(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    times = [
        "2026-06-01 12:00:00",
        "2026-06-01 12:01:00",
        "2026-06-01 12:03:00",  # session 0: 180 s, electrodes {0, 1}
        "2026-06-01 14:03:00",  # 2-hour gap
        "2026-06-01 14:05:30",  # session 1: 150 s, electrodes {2, 3, 4}
        "2026-06-01 14:04:00",  # unsorted on disk
    ]
    electrodes = [0, 1, 0, 2, 3, 4]
    _write_finalspark_events(path, times, electrodes)

    # ACT
    probes, a = probe(
        _entry(path, "finalspark"),
        census_skip_s=SKIP_S,
        census_window_len_s=WINDOW_S,
        finalspark_session_gap_s=GAP_S,
    )

    # ASSERT
    assert len(probes) == 2
    assert [p.session_index for p in probes] == [0, 1]
    assert probes[0].duration_s == pytest.approx(180.0)
    assert probes[1].duration_s == pytest.approx(150.0)
    assert probes[0].n_channels == 2
    assert probes[1].n_channels == 3
    assert probes[0].recording_timestamp == "2026-06-01T12:00:00"
    assert probes[1].recording_timestamp == "2026-06-01T14:03:00"
    assert probes[0].recording_id != probes[1].recording_id
    assert all(len(p.recording_id) == 12 for p in probes)
    assert all(p.probe_error is None for p in probes)
    assert all(p.has_spike_table for p in probes)
    assert a["n_sessions"] == 2
    assert a["n_events"] == 6
    assert a["events_read_method"] == "h5py_fields"


def test_probe_finalspark_short_session_flagged(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs1_package.hdf5"
    _write_finalspark_events(path, ["2026-06-01 12:00:00", "2026-06-01 12:01:00"], [0, 1])

    # ACT
    p, _ = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.duration_s == pytest.approx(60.0)
    assert p.session_index == 0
    assert p.probe_error is not None and "census_window_len_s" in p.probe_error


def test_probe_finalspark_xy_absent_is_verified_and_auditable(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    _write_finalspark_events(
        path, ["2026-06-01 12:00:00", "2026-06-01 12:05:00"], [0, 1]
    )
    meta = pd.DataFrame({"organoid_id": ["fs437"], "layout": ["4x8"]})
    meta.to_hdf(path, key="fs437_wholelife_metadata", mode="a")

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.has_xy is False
    assert p.xy_units == "absent"
    assert p.xy_provenance == "absent_verified"
    assert a["metadata_fields"] == {"fs437_wholelife_metadata": ["organoid_id", "layout"]}
    assert "time_of_event" in a["events_columns"]
    assert "fs437_wholelife_events/table" in a["visited_keys"]
    assert "fs437_wholelife_metadata/axis0" in a["visited_keys"]


def test_probe_finalspark_xy_in_events_columns(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    events = pd.DataFrame(
        {
            "electrode": [0, 1],
            "time_of_event": pd.to_datetime(["2026-06-01 12:00:00", "2026-06-01 12:05:00"]),
            "electrode_x": [1.0, 2.0],
            "electrode_y": [3.0, 4.0],
        }
    )
    events.to_hdf(
        path, key="fs437_wholelife_events", mode="w", format="table",
        data_columns=["time_of_event", "electrode"],
    )

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.has_xy is True
    assert p.xy_provenance == "events_table_columns"
    assert p.xy_units == "unknown"
    assert a["xy_columns"] == ["electrode_x", "electrode_y"]


def test_probe_finalspark_xy_in_metadata(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    _write_finalspark_events(path, ["2026-06-01 12:00:00", "2026-06-01 12:05:00"], [0, 1])
    meta = pd.DataFrame({"electrode": [0, 1], "pos_x": [0.0, 1.0], "pos_y": [0.0, 1.0]})
    meta.to_hdf(path, key="fs437_wholelife_metadata", mode="a")

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.has_xy is True
    assert p.xy_provenance == "metadata_table"
    assert a["xy_source_key"] == "fs437_wholelife_metadata"


def test_probe_finalspark_xy_found_by_tree_walk(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    _write_finalspark_events(path, ["2026-06-01 12:00:00", "2026-06-01 12:05:00"], [0, 1])
    with h5py.File(path, "a") as f:
        dt = np.dtype([("electrode", "<i8"), ("x", "<f8"), ("y", "<f8")])
        f.create_dataset("extra/deep/layout", data=np.zeros(2, dtype=dt))

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.has_xy is True
    assert p.xy_provenance == "mapping_dataset"
    assert a["xy_source_key"] == "extra/deep/layout"


def test_probe_finalspark_stim_and_env_tables(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    _write_finalspark_events(path, ["2026-06-01 12:00:00", "2026-06-01 12:05:00"], [0, 1])
    stim = pd.DataFrame(
        {
            "electrode": [0, 1, 2],
            "time_of_stim": pd.to_datetime(
                ["2026-06-01 12:02:00", "2026-06-01 11:00:00", "2026-06-02 08:00:00"]
            ),
            "a1": [1.0, 2.0, 3.0],
        }
    )
    stim.to_hdf(
        path, key="fs437_wholelife_stimulations", mode="a", format="table",
        data_columns=["time_of_stim", "electrode"],
    )
    for name in ["CO2", "O2", "temperature", "door_opening"]:
        env = pd.DataFrame(
            {"time": pd.to_datetime(["2026-06-01 12:00:00"]), "value": [1.0]}
        )
        env.to_hdf(path, key=f"fs437_wholelife_incubator_{name}", mode="a", format="table")

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.has_stim_table is True
    assert p.n_stim_events == 3
    assert p.stim_time_min == "2026-06-01T11:00:00"
    assert p.stim_time_max == "2026-06-02T08:00:00"
    assert p.has_env_channels is True
    assert sorted(p.env_channel_names) == [
        "fs437_wholelife_incubator_CO2",
        "fs437_wholelife_incubator_O2",
        "fs437_wholelife_incubator_door_opening",
        "fs437_wholelife_incubator_temperature",
    ]
    assert a["stim_read_method"] == "h5py_fields"


def test_probe_finalspark_without_stim_or_env(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    _write_finalspark_events(path, ["2026-06-01 12:00:00", "2026-06-01 12:05:00"], [0, 1])

    # ACT
    p, _ = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.has_stim_table is False
    assert p.n_stim_events is None
    assert p.stim_time_min is None and p.stim_time_max is None
    assert p.has_env_channels is False
    assert p.env_channel_names == []


def test_probe_finalspark_empty_events_sets_error(tmp_path: Path):
    # ARRANGE: pandas does not write empty table-format frames, so build the
    # PyTables-style layout directly.
    path = tmp_path / "fs437_package.hdf5"
    dt = np.dtype([("index", "<i8"), ("time_of_event", "<i8"), ("electrode", "<i8")])
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("fs437_wholelife_events/table", shape=(0,), dtype=dt)
        ds.attrs["time_of_event_dtype"] = np.bytes_(b"datetime64[ns]")

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.probe_error is not None and "is empty" in p.probe_error
    assert p.duration_s is None
    assert p.session_index is None
    assert a["n_sessions"] == 0


def test_probe_finalspark_without_events_table(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "fs437_package.hdf5"
    pd.DataFrame({"a": [1]}).to_hdf(path, key="fs437_wholelife_metadata", mode="w")

    # ACT
    p, a = _probe_one(_entry(path, "finalspark"))

    # ASSERT
    assert p.probe_error is not None and "No events table" in p.probe_error
    assert p.xy_units == "absent"
    assert "error" in a


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_probe_handles_malformed_file(tmp_path: Path):
    # ARRANGE
    corrupt = tmp_path / "corrupt.raw.h5"
    corrupt.write_bytes(b"THIS_IS_NOT_A_VALID_HDF5_FILE")

    # ACT
    p, a = _probe_one(_entry(corrupt, "maxwell"))

    # ASSERT
    assert p.probe_error is not None
    assert p.has_xy is False
    assert p.xy_units == "absent"
    assert p.has_spike_table is False
    assert "error" in a


def test_probe_unknown_source(tmp_path: Path):
    # ARRANGE
    path = tmp_path / "unknown.dat"
    path.write_bytes(b"dummy")

    # ACT
    p, a = _probe_one(_entry(path, "unknown_source"))

    # ASSERT
    assert "Unknown source identifier" in p.probe_error
    assert p.xy_provenance == "absent"
    assert "error" in a


# ---------------------------------------------------------------------------
# run_census_probe
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, roots: dict, output_dir: Path, drop: str | None = None) -> Path:
    cfg = {
        "roots": roots,
        "patterns": {s: ALL_PATTERNS[s] for s in roots},
        "output_dir": str(output_dir),
        "census_skip_s": SKIP_S,
        "census_window_len_s": WINDOW_S,
        "finalspark_session_gap_s": GAP_S,
    }
    if drop is not None:
        cfg.pop(drop)
    path = tmp_path / "recovery_census.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_run_census_probe_pipeline_one_row_per_session(tmp_path: Path):
    # ARRANGE
    mx_dir = tmp_path / "maxwell"
    fs_dir = tmp_path / "finalspark"
    mx_dir.mkdir()
    fs_dir.mkdir()
    output_dir = tmp_path / "output"
    _write_maxwell_v1(mx_dir / "sample.raw.h5", 4, 300, sampling_rate=2.5)
    _write_finalspark_events(
        fs_dir / "fs9_package.hdf5",
        ["2026-06-01 12:00:00", "2026-06-01 12:05:00", "2026-06-01 15:00:00"],
        [0, 1, 2],
    )
    config_path = _write_config(
        tmp_path, {"maxwell": str(mx_dir), "finalspark": str(fs_dir)}, output_dir
    )

    # ACT
    df, summary = run_census_probe(str(config_path))

    # ASSERT
    assert len(df) == 3
    assert summary["total_files"] == 2
    assert summary["total_probes"] == 3
    assert summary["sources"]["finalspark"]["probe_count"] == 2
    assert df["xy_units"].notna().all()
    assert os.path.exists(output_dir / "probe_table.csv")
    csv = pd.read_csv(output_dir / "probe_table.csv")
    assert "session_index" in csv.columns and "xy_units" in csv.columns
    with open(output_dir / "probe_assumptions.json") as f:
        saved = json.load(f)
    assert saved["finalspark_session_gap_s"] == GAP_S
    assert saved["sources"]["maxwell"]["file_count"] == 1


def test_run_census_probe_requires_window_len(tmp_path: Path):
    # ARRANGE
    config_path = _write_config(
        tmp_path, {"maxwell": str(tmp_path)}, tmp_path / "out", drop="census_window_len_s"
    )

    # ACT / ASSERT
    with pytest.raises(KeyError, match="census_window_len_s"):
        run_census_probe(str(config_path))
