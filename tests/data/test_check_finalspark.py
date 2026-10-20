"""Tests for src/data/ephys/check_finalspark.py using synthetic HDF5 fixtures."""

import pandas as pd
import pytest

from src.data.ephys.check_finalspark import check_package, find_package


def _write_synthetic_package(path, fs_id: str) -> None:
    """Write a minimal synthetic FinalSpark package HDF5 to `path`."""
    stim = pd.DataFrame(
        {
            "electrode": [0, 0, 1],
            "time_of_stim": pd.to_datetime(
                ["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:03"]
            ),
            "a1": [1.0, 2.0, 1.0],
        }
    )
    events = pd.DataFrame({"electrode": [0, 1, 2], "time_of_event": [0.0, 1.0, 2.0]})
    meta = pd.DataFrame({"Title": [fs_id], "Number of NS": [4.0]})
    stim.to_hdf(path, key=f"{fs_id}_wholelife_stimulations", format="table")
    events.to_hdf(path, key=f"{fs_id}_wholelife_events", format="table")
    meta.to_hdf(path, key=f"{fs_id}_wholelife_metadata")


def test_find_package_infers_culture_id(tmp_path):
    # ARRANGE
    pkg_path = tmp_path / "fs999_package.hdf5"
    _write_synthetic_package(pkg_path, "fs999")

    # ACT
    pkg, fs_id = find_package(str(tmp_path))

    # ASSERT
    assert pkg == str(pkg_path)
    assert fs_id == "fs999"


def test_find_package_missing_raises(tmp_path):
    # ARRANGE
    empty_dir = tmp_path

    # ACT / ASSERT
    with pytest.raises(FileNotFoundError):
        find_package(str(empty_dir))


def test_find_package_multiple_raises(tmp_path):
    # ARRANGE
    _write_synthetic_package(tmp_path / "fs1_package.hdf5", "fs1")
    _write_synthetic_package(tmp_path / "fs2_package.hdf5", "fs2")

    # ACT / ASSERT
    with pytest.raises(ValueError):
        find_package(str(tmp_path))


def test_check_package_reports_never_stimulated(tmp_path, capsys):
    # ARRANGE
    pkg_path = tmp_path / "fs999_package.hdf5"
    _write_synthetic_package(pkg_path, "fs999")
    expected_never = [2]

    # ACT
    summary = check_package(str(pkg_path), "fs999")
    out = capsys.readouterr().out

    # ASSERT
    assert summary["n_stim"] == 3
    assert summary["stim_electrodes"] == [0, 1]
    assert summary["event_electrodes"] == [0, 1, 2]
    assert summary["never_stimulated"] == expected_never
    assert "export_state: not present in store" in out
    assert "=== INTER-STIM INTERVALS (s) ===" in out
