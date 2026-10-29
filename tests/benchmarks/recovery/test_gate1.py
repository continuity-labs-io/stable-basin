"""Unit tests for the Gate 1 analysis functions (src/benchmarks/recovery/gate1.py).

All inputs are synthetic and built in the test; no HDF5 or JSON fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmarks.recovery.gate1 import (
    analyse,
    centered_r,
    latencies,
    measures,
    pearson,
    spearman_brown,
    sweep_blocks,
)

EDGES = np.geomspace(0.003, 10.0, 41)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
RETURN_X, HALF_FRAC, RESIDUAL_S = 1.5, 0.5, 1.0
TUPLE = ["a1", "stim_polarity"]
T0 = pd.Timestamp("2025-02-20 16:00:00")


def _measures(trials, baseline):
    return measures(trials, baseline, EDGES, RETURN_X, HALF_FRAC, RESIDUAL_S)


def _stim_rows(blocks):
    """Build stimulation rows: blocks is a list of (a1, n_electrodes, n_stims); stimuli 10 s apart."""
    rows, t = [], 0
    for a1, n_elec, n_stims in blocks:
        for _ in range(n_stims):
            for e in range(n_elec):
                rows.append(dict(time_of_stim=T0 + pd.Timedelta(seconds=10 * t), electrode=e, a1=a1, stim_polarity=0))
            t += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- measures


def test_measures_t_return_is_centre_of_last_bin_above_threshold():
    # ARRANGE: one event per trial in bin 10, nothing else; baseline low enough that only bin 10 exceeds 1.5x.
    k, baseline = 10, 0.01
    trials = [np.array([CENTERS[k]]), np.array([CENTERS[k]])]

    # ACT
    out = _measures(trials, baseline)

    # ASSERT
    assert out["t_return"] == pytest.approx(CENTERS[k])
    assert bool(out["t_return_censored"]) is False
    assert out["n_trials"] == 2
    assert out["n_events"] == 2


def test_measures_t_return_zero_when_no_bin_exceeds_threshold():
    # ARRANGE: no events at all.
    trials = [np.array([]), np.array([])]

    # ACT
    out = _measures(trials, baseline=1.0)

    # ASSERT
    assert out["t_return"] == 0.0
    assert bool(out["t_return_censored"]) is False
    assert out["residual"] == 0.0


def test_measures_t_return_censored_when_final_bin_exceeds_threshold():
    # ARRANGE: one event per trial at 9.5 s (final bin, and inside the last 1 s residual window).
    baseline = 0.1
    trials = [np.array([9.5]), np.array([9.5])]

    # ACT
    out = _measures(trials, baseline)

    # ASSERT: censored at the window end; residual = (2 events / (2 trials * 1 s)) / 0.1 Hz = 10.
    assert out["t_return"] == 10.0
    assert bool(out["t_return_censored"]) is True
    assert out["residual"] == pytest.approx(10.0)


def test_measures_t_half_peak_is_time_to_first_bin_at_or_below_half_peak():
    # ARRANGE: 4 events per trial in bin 10 (peak), none in bin 11.
    k = 10
    trials = [np.full(4, CENTERS[k]), np.full(4, CENTERS[k])]

    # ACT
    out = _measures(trials, baseline=1.0)

    # ASSERT
    assert out["t_peak"] == pytest.approx(CENTERS[k])
    assert out["t_half_peak"] == pytest.approx(CENTERS[k + 1] - CENTERS[k])
    assert bool(out["t_half_peak_censored"]) is False


def test_measures_t_half_peak_censored_when_peak_is_in_final_bin():
    # ARRANGE: peak in the final bin, so no later bin can fall to half.
    trials = [np.array([9.5]), np.array([9.5])]

    # ACT
    out = _measures(trials, baseline=0.1)

    # ASSERT
    assert out["t_half_peak"] == 10.0
    assert bool(out["t_half_peak_censored"]) is True


# ---------------------------------------------------------------- correlation helpers


def test_spearman_brown_known_values():
    # ARRANGE
    inputs, expected = [0.0, 0.5, 1.0], [0.0, 2 / 3, 1.0]

    # ACT
    out = [spearman_brown(r) for r in inputs]

    # ASSERT
    assert out == pytest.approx(expected)


def test_pearson_matches_numpy_and_flattens():
    # ARRANGE
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(4, 5)), rng.normal(size=(4, 5))
    expected = np.corrcoef(x.ravel(), y.ravel())[0, 1]

    # ACT
    out = pearson(x, y)

    # ASSERT
    assert out == pytest.approx(expected)


def test_pearson_nan_when_input_constant():
    # ARRANGE
    x, y = np.ones(5), np.arange(5.0)

    # ACT
    out = pearson(x, y)

    # ASSERT
    assert np.isnan(out)


def test_centered_r_invariant_to_per_block_offsets():
    # ARRANGE: adding a constant to each block (column) on either day must not change the result.
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    offsets_a, offsets_b = rng.normal(size=4) * 100, rng.normal(size=4) * 100

    # ACT
    r_plain = centered_r(a, b)
    r_offset = centered_r(a + offsets_a, b + offsets_b)

    # ASSERT
    assert r_offset == pytest.approx(r_plain)


def test_centered_r_raises_on_shape_mismatch():
    # ARRANGE
    a, b = np.zeros((6, 4)), np.zeros((5, 4))

    # ACT / ASSERT
    with pytest.raises(ValueError, match="shape mismatch"):
        centered_r(a, b)


# ---------------------------------------------------------------- sweep_blocks


def test_sweep_blocks_drops_blocks_below_min_electrodes():
    # ARRANGE: block 0 hits 1 electrode, blocks 1 and 2 hit 3.
    rows = _stim_rows([(2.5, 1, 3), (0.8, 3, 2), (1.0, 3, 2)])

    # ACT
    kept, times, elec, n_blocks = sweep_blocks(rows, TUPLE, min_electrodes=3)

    # ASSERT
    assert n_blocks == 3
    assert kept["block"].tolist() == [1, 2]
    assert kept["a1"].tolist() == [0.8, 1.0]
    assert [len(times[b]) for b in kept["block"]] == [2, 2]
    assert elec.tolist() == [0, 1, 2]


def test_sweep_blocks_raises_when_one_stimulus_has_two_tuples():
    # ARRANGE: same time_of_stim, two different a1 values.
    rows = pd.DataFrame(dict(time_of_stim=[T0, T0], electrode=[0, 1], a1=[0.8, 1.0], stim_polarity=[0, 0]))

    # ACT / ASSERT
    with pytest.raises(ValueError, match="more than one parameter tuple"):
        sweep_blocks(rows, TUPLE, min_electrodes=1)


def test_sweep_blocks_raises_when_tuple_repeats_across_kept_blocks():
    # ARRANGE: a1 sequence 0.8, 1.0, 0.8 gives two separate blocks with the same tuple.
    rows = _stim_rows([(0.8, 2, 2), (1.0, 2, 2), (0.8, 2, 2)])

    # ACT / ASSERT
    with pytest.raises(ValueError, match="repeats across sweep blocks"):
        sweep_blocks(rows, TUPLE, min_electrodes=2)


# ---------------------------------------------------------------- latencies


def test_latencies_window_is_lower_inclusive_upper_exclusive():
    # ARRANGE: events at +0.001 s (before window), +0.003 s (at lo), +0.5 s, +10.0 s (at hi, excluded).
    stim = np.array([T0.to_datetime64()])
    offsets = [0.001, 0.003, 0.5, 10.0]
    ev = pd.DataFrame(dict(electrode=[7] * 4, time_of_event=[T0 + pd.Timedelta(seconds=s) for s in offsets]))

    # ACT
    out = latencies(ev, [7], [stim], post_lo_s=0.003, post_hi_s=10.0)

    # ASSERT
    assert list(out) == [(7, 0)]
    assert len(out[7, 0]) == 1
    np.testing.assert_allclose(out[7, 0][0], [0.003, 0.5])


# ---------------------------------------------------------------- analyse


def _analyse_inputs(rng, same_electrode_effect):
    """Day matrices = electrode effect + block effect + noise; halves = day matrix + extra noise."""
    n_e, n_b, days = 20, 8, ["d1", "d2"]
    blocks = rng.normal(size=n_b) * 5
    e1 = rng.normal(size=(n_e, 1))
    e2 = e1 if same_electrode_effect else rng.normal(size=(n_e, 1))
    mat = {days[0]: e1 + blocks + 0.05 * rng.normal(size=(n_e, n_b)),
           days[1]: e2 + blocks + 0.05 * rng.normal(size=(n_e, n_b))}
    half = {(d, s): mat[d] + 0.05 * rng.normal(size=(n_e, n_b)) for d in days for s in ("odd", "even")}
    sym = np.array([True] * 4 + [False] * 4)
    return mat, half, days, np.array([0.8, 1.0, 1.2, 1.5]), sym


def test_analyse_recovers_shared_electrode_effect():
    # ARRANGE
    rng = np.random.default_rng(3)
    mat, half, days, sym_a1, sym = _analyse_inputs(rng, same_electrode_effect=True)

    # ACT
    res = analyse(mat, half, days, sym_a1, sym, rng, n_perm=200, null_pctl=95)

    # ASSERT
    assert res["condition_centered_r"] > 0.9
    assert res["condition_centered_r"] > res["null_p95"]
    assert res["null_frac_ge_observed"] == 0.0
    assert 0 < res["ceiling"] <= 1
    assert len(res["per_block_r"]) == 8


def test_analyse_finds_no_effect_when_electrode_effects_independent():
    # ARRANGE
    rng = np.random.default_rng(4)
    mat, half, days, sym_a1, sym = _analyse_inputs(rng, same_electrode_effect=False)

    # ACT
    res = analyse(mat, half, days, sym_a1, sym, rng, n_perm=200, null_pctl=95)

    # ASSERT: high within-day ceiling, but no cross-day electrode effect beyond the null.
    assert res["ceiling"] > 0.9
    assert res["condition_centered_r"] < res["null_p95"]
    assert res["null_frac_ge_observed"] > 0.01
