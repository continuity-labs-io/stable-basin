"""Tests for src/data/ephys/finalspark_protocol.py using synthetic data.

All stimulation tables and HDF5 packages are generated inside the tests.
"""

import numpy as np
import pandas as pd
import pytest

import src.data.ephys.finalspark_protocol as fp

T0 = pd.Timestamp("2025-01-01T00:00:00")


def _stim(times_s, electrodes=None, **cols) -> pd.DataFrame:
    """Build a synthetic stimulation table.

    Args:
        times_s: Seconds after T0 for each row.
        electrodes: Electrode per row (default all 0).
        **cols: Overrides for parameter columns (scalar or per-row list).

    Returns:
        DataFrame with `electrode`, `time_of_stim`, and all `PARAM_COLS`.
    """
    n = len(times_s)
    defaults = {
        "a1": 1.0,
        "a2": 1.0,
        "d1": 300.0,
        "d2": 300.0,
        "nb_stim_pulse": 2.0,
        "pulse_train_period": 10000.0,
        "stim_polarity": 0.0,
        "stim_shape": 0.0,
    }
    defaults.update(cols)
    df = pd.DataFrame(
        {
            "electrode": electrodes if electrodes is not None else [0] * n,
            "time_of_stim": T0 + pd.to_timedelta(times_s, unit="s"),
        }
    )
    for c, v in defaults.items():
        df[c] = v
    return df


# ---------------------------------------------------------------- TASK 1


def test_assign_groups_splits_on_threshold_and_electrode():
    # ARRANGE
    stim = _stim([0.0, 0.01, 1.0, 0.0, 0.01], electrodes=[0, 0, 0, 1, 1])
    expected_groups = [0, 0, 1, 2, 2]

    # ACT
    out = fp.assign_groups(stim, group_gap_s=0.08)

    # ASSERT
    assert out["group_id"].tolist() == expected_groups
    assert np.isnan(out["gap_s"].iloc[0])
    assert np.isnan(out["gap_s"].iloc[3])  # first row of electrode 1
    assert out["gap_s"].iloc[1] == pytest.approx(0.01)


def test_assign_groups_rejects_nonpositive_threshold():
    # ARRANGE
    stim = _stim([0.0, 1.0])

    # ACT / ASSERT
    with pytest.raises(ValueError):
        fp.assign_groups(stim, group_gap_s=0.0)


def test_row_semantics_pulses_when_group_sizes_match():
    # ARRANGE: 3 trains of 2 pulses 10 ms apart, trains 10 s apart.
    times = [0.0, 0.01, 10.0, 10.01, 20.0, 20.01]
    stim = _stim(times, nb_stim_pulse=2.0)

    # ACT
    out = fp.row_semantics(stim, group_gap_s=0.08, match_fraction=0.9)

    # ASSERT
    assert out["median_size"] == {2.0: 2.0}
    assert out["match_frac"] == pytest.approx(1.0)
    assert out["rows_are_pulses"] is True
    assert out["verdict"] == "ROWS ARE PULSES"


def test_row_semantics_triggers_when_rows_isolated():
    # ARRANGE: every row 1 s apart, nb_stim_pulse=2 -> groups of size 1.
    stim = _stim([0.0, 1.0, 2.0, 3.0], nb_stim_pulse=2.0)

    # ACT
    out = fp.row_semantics(stim, group_gap_s=0.08, match_fraction=0.9)

    # ASSERT
    assert out["median_size"] == {2.0: 1.0}
    assert out["match_frac"] == 0.0
    assert out["verdict"] == "ROWS ARE TRIGGERS"


# ---------------------------------------------------------------- TASK 2


@pytest.mark.parametrize(
    "ratio,expected",
    [(1.1e-6, "microseconds"), (0.9e-3, "milliseconds"), (1.2, "seconds"), (0.0, "undetermined")],
)
def test_implied_unit(ratio, expected):
    # ARRANGE
    r = ratio

    # ACT
    unit = fp.implied_unit(r)

    # ASSERT
    assert unit == expected


def test_train_period_units_observable_reports_microseconds():
    # ARRANGE: pulses 10 ms apart with pulse_train_period=10000 -> 1e-6 ratio.
    stim = _stim([0.0, 0.01, 10.0, 10.01], pulse_train_period=10000.0)
    t1 = fp.row_semantics(stim, group_gap_s=0.08, match_fraction=0.9)

    # ACT
    out = fp.train_period_units(t1)

    # ASSERT
    assert out["observable"] is True
    obs, ratio, unit = out["ratios"][10000.0]
    assert obs == pytest.approx(0.01)
    assert ratio == pytest.approx(1e-6)
    assert unit == "microseconds"


def test_train_period_units_unobservable_gives_gap_histogram():
    # ARRANGE
    stim = _stim([0.0, 1.5, 3.0], pulse_train_period=[1.0, 10000.0, 10000.0])
    t1 = fp.row_semantics(stim, group_gap_s=0.08, match_fraction=0.9)

    # ACT
    out = fp.train_period_units(t1)

    # ASSERT
    assert out["observable"] is False
    assert "ratios" not in out
    assert int(out["gap_hist"].to_numpy().sum()) == 2


# ---------------------------------------------------------------- TASK 3


def test_dose_vs_age_confounded_when_a1_rises_with_days():
    # ARRANGE: a1 increases monotonically over 4 days.
    day = 86400.0
    stim = _stim([0.0, day, 2 * day, 3 * day], a1=[1.0, 2.0, 3.0, 4.0])

    # ACT
    out = fp.dose_vs_age(stim, rho_threshold=0.3)

    # ASSERT
    assert out["rho"] == pytest.approx(1.0)
    assert out["verdict"] == "DOSE CONFOUNDED WITH AGE"
    assert out["daily"].index.tolist() == [0, 1, 2, 3]
    assert out["daily"]["n_stims"].tolist() == [1, 1, 1, 1]


def test_dose_vs_age_usable_when_a1_alternates():
    # ARRANGE: a1 pattern with zero rank correlation to time.
    day = 86400.0
    stim = _stim([0.0, day, 2 * day, 3 * day], a1=[1.0, 2.0, 2.0, 1.0])

    # ACT
    out = fp.dose_vs_age(stim, rho_threshold=0.3)

    # ASSERT
    assert abs(out["rho"]) < 0.3
    assert out["verdict"] == "DOSE AXIS USABLE"
    assert out["daily"].loc[0, "d1_values"] == [300.0]
    assert out["daily"].loc[0, "iqr_a1"] == 0.0


# ---------------------------------------------------------------- TASK 4


def test_parameter_sets_counts_and_widest():
    # ARRANGE: set A (a1=1) on 1 day x3 rows; set B (a1=2) on 2 days, 2 electrodes.
    day = 86400.0
    stim = _stim(
        [0.0, 1.0, 2.0, 3.0, day + 3.0],
        electrodes=[0, 0, 0, 1, 2],
        a1=[1.0, 1.0, 1.0, 2.0, 2.0],
    )

    # ACT
    out = fp.parameter_sets(stim)

    # ASSERT
    assert out["n_sets"] == 2
    assert out["sets"]["n_stims"].tolist() == [3, 2]  # sorted by count
    assert out["widest"][0] == 2.0
    assert int(out["sets"].loc[out["widest"], "n_days"]) == 2
    assert int(out["sets"].loc[out["widest"], "n_electrodes"]) == 2


# ---------------------------------------------------------------- TASK 5


def test_unstimulated_epochs_baseline_gaps_tail():
    # ARRANGE: stims at 1 h, 2 h, 12 h (10 h gap); events from 0 h to 14 h.
    stim = _stim([3600.0, 7200.0, 12 * 3600.0])
    first_event = T0
    last_event = T0 + pd.Timedelta(hours=14)

    # ACT
    out = fp.unstimulated_epochs(stim, first_event, last_event, long_gap_h=6.0)

    # ASSERT
    assert out["baseline_h"] == pytest.approx(1.0)
    assert out["tail_h"] == pytest.approx(2.0)
    assert len(out["long_gaps"]) == 1
    assert out["long_gaps"]["duration_h"].iloc[0] == pytest.approx(10.0)
    assert out["total_h"] == pytest.approx(13.0)


def test_unstimulated_epochs_clips_negative_baseline():
    # ARRANGE: stimulation starts before the first event.
    stim = _stim([0.0, 3600.0])
    first_event = T0 + pd.Timedelta(hours=2)
    last_event = T0 + pd.Timedelta(hours=2)

    # ACT
    out = fp.unstimulated_epochs(stim, first_event, last_event, long_gap_h=6.0)

    # ASSERT
    assert out["baseline_h"] < 0
    assert out["long_gaps"].empty
    assert out["total_h"] == pytest.approx(1.0)  # only the 1 h tail counts


# ---------------------------------------------------------------- I/O + report


def test_load_tables_scans_events_in_chunks(tmp_path, monkeypatch):
    # ARRANGE: unsorted event times; chunk size smaller than the table.
    fs_id = "fs999"
    pkg = tmp_path / f"{fs_id}_package.hdf5"
    _stim([0.0, 1.0]).to_hdf(pkg, key=f"{fs_id}_wholelife_stimulations", format="table")
    ev_times = T0 + pd.to_timedelta([50.0, -5.0, 10.0, 99.0, 20.0], unit="s")
    events = pd.DataFrame({"electrode": [0] * 5, "time_of_event": ev_times})
    events.to_hdf(pkg, key=f"{fs_id}_wholelife_events", format="table", data_columns=True)
    monkeypatch.setattr(fp, "EVENT_CHUNK_ROWS", 2)

    # ACT
    stim, first_event, last_event = fp.load_tables(str(pkg), fs_id)

    # ASSERT
    assert len(stim) == 2
    assert first_event == T0 - pd.Timedelta(seconds=5)
    assert last_event == T0 + pd.Timedelta(seconds=99)


def test_run_report_prints_five_verdict_lines(capsys):
    # ARRANGE
    day = 86400.0
    stim = _stim([0.0, 1.0, day, day + 1.0], a1=[1.0, 1.0, 2.0, 2.0])
    first_event = T0 - pd.Timedelta(hours=1)
    last_event = T0 + pd.Timedelta(days=2)
    expected_prefixes = ["ROWS ARE", "UNOBSERVABLE", "DOSE", "WIDEST SET", "UNSTIMULATED"]

    # ACT
    summary = fp.run_report(stim, first_event, last_event)
    out = capsys.readouterr().out

    # ASSERT
    assert len(summary) == 5
    for line, prefix in zip(summary, expected_prefixes):
        assert line.startswith(prefix)
    assert out.rstrip().splitlines()[-5:] == summary
