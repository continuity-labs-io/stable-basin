import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def compute_stats(y, o):
    """
    Computes statistical significance metrics between two cohorts.
    Returns a dictionary of statistics including Welch's t-test,
    Mann-Whitney U test, and Cohen's d effect size.
    """
    t_stat, t_p = ttest_ind(y, o, equal_var=False)
    u_stat, u_p = mannwhitneyu(y, o)
    nx = len(y)
    ny = len(o)
    dof = nx + ny - 2
    if dof <= 0:
        return {"welch_t_stat": 0.0, "welch_p_value": 1.0, "mann_whitney_u_stat": 0.0, "mann_whitney_p_value": 1.0, "cohens_d": 0.0}
    pool_sd = np.sqrt(((nx - 1) * np.var(y, ddof=1) + (ny - 1) * np.var(o, ddof=1)) / dof)
    d = (np.mean(y) - np.mean(o)) / pool_sd if pool_sd > 0 else 0.0
    return {
        "welch_t_stat": float(t_stat),
        "welch_p_value": float(t_p),
        "mann_whitney_u_stat": float(u_stat),
        "mann_whitney_p_value": float(u_p),
        "cohens_d": float(d)
    }
