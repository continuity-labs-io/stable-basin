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
