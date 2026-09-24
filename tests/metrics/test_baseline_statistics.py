import numpy as np
import pytest
from src.metrics.baseline_statistics import compute_stats


def test_compute_stats_baseline():
    """
    Test the statistical significance metrics computation.
    Must adhere to ARRANGE, ACT, ASSERT block structure.
    """
    # ARRANGE
    np.random.seed(42)
    young_cohort = np.random.normal(loc=0.5, scale=0.1, size=100)
    old_cohort = np.random.normal(loc=0.8, scale=0.4, size=100)

    # ACT
    stats = compute_stats(young_cohort, old_cohort)

    # ASSERT
    assert "welch_t_stat" in stats
    assert "welch_p_value" in stats
    assert "mann_whitney_u_stat" in stats
    assert "mann_whitney_p_value" in stats
    assert "cohens_d" in stats

    # Since old cohort has a higher mean and higher variance
    # Cohen's d should be negative (young - old < 0)
    assert stats["cohens_d"] < 0
    # p-values should be very small given the large mean difference
    assert stats["welch_p_value"] < 0.05
    assert stats["mann_whitney_p_value"] < 0.05

def test_hedges_g():
    """Test small-sample corrected effect size."""
    from src.metrics.baseline_statistics import hedges_g
    # ARRANGE
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([3.0, 4.0, 5.0, 6.0])
    # ACT
    g = hedges_g(a, b)
    # ASSERT
    assert g > 0.0
    assert not np.isnan(g)

def test_unpaired_stats():
    """Test unpaired permutation tests."""
    from src.metrics.baseline_statistics import unpaired_stats
    # ARRANGE
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20)
    b = rng.normal(1, 1, 20)
    # ACT
    stats = unpaired_stats(a, b, rng, n_perm=100, n_boot=100)
    # ASSERT
    assert "perm_p" in stats
    assert "hedges_g" in stats
    assert stats["mean_diff"] > 0

def test_paired_stats():
    """Test paired significance tests."""
    from src.metrics.baseline_statistics import paired_stats
    # ARRANGE
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, 20)
    degraded = clean + 1.0  # Paired shift
    # ACT
    stats = paired_stats(clean, degraded, rng, n_perm=100, n_boot=100)
    # ASSERT
    assert "signflip_p" in stats
    assert "cohens_dz" in stats
    assert stats["mean_delta"] > 0

def test_naive_timestep_ks():
    """Test timestep KS aggregation."""
    from src.metrics.baseline_statistics import naive_timestep_ks
    # ARRANGE
    a = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    b = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    # ACT
    stats = naive_timestep_ks(a, b)
    # ASSERT
    assert stats["p"] < 1.0
    assert "note" in stats

def test_resplit_stats():
    """Test resplit for FPR evaluation."""
    from src.metrics.baseline_statistics import resplit_stats
    # ARRANGE
    clean = np.random.normal(0, 1, 20)
    cond = np.random.normal(0, 1, 20)
    labels = np.array([0]*10 + [1]*10)
    # ACT
    stats = resplit_stats(clean, cond, labels, n_splits=5, seed=42)
    # ASSERT
    assert stats["n_splits"] == 5
    assert "frac_p_lt_0.05" in stats
