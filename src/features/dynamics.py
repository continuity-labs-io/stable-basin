import pandas as pd
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
import logging

logger = logging.getLogger(__name__)

def calculate_iact(acf_norm: np.ndarray, var_xf: np.ndarray, n: int, num_vars: int) -> np.ndarray:
    """Calculates Integrated Autocorrelation Time (IACT) from normalized ACF."""
    iact = np.zeros(num_vars)
    max_lags = min(30, n)
    for f_idx in range(num_vars):
        if var_xf[f_idx] == 0:
            continue
        acf_sum = 0
        for k in range(1, max_lags):
            if acf_norm[k, f_idx] <= 0:
                break
            acf_sum += acf_norm[k, f_idx]
        iact[f_idx] = 1 + 2 * acf_sum
    return iact

def compute_f2a_features(daily_means: pd.DataFrame, variables: list[str]) -> tuple[np.ndarray, list[str]]:
    """Computes macroscopic F2a features (variance, lag-1 ACF, IACT)."""
    X = daily_means.values
    t = daily_means.index.values
    mean_X = np.mean(X, axis=0)
    var_X = np.var(X, axis=0)
    
    t_center = t - np.mean(t)
    var_t = np.sum(t_center ** 2)
    
    num_vars = len(variables)
    
    if len(X) > 1 and var_t > 0:
        cov = np.sum(t_center[:, None] * (X - mean_X), axis=0)
        slope = cov / var_t
        X_detrended = X - mean_X - slope * t_center[:, None]
        
        n = X_detrended.shape[0]
        N = next_fast_len(2 * n)
        F = rfft(X_detrended, n=N, axis=0)
        acf = irfft(F * np.conj(F), n=N, axis=0)[:n]
        
        var_xf = np.sum(X_detrended ** 2, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            acf_norm = acf / var_xf
            acf_norm[:, var_xf == 0] = 0
            
        lag1_acf = acf_norm[1] if n > 1 else np.zeros(num_vars)
        iact = calculate_iact(acf_norm, var_xf, n, num_vars)
    else:
        logger.debug("compute_f2a_features fallback: insufficient data or var_t == 0, returning zeros for ACF and IACT.")
        lag1_acf = np.zeros(num_vars)
        iact = np.zeros(num_vars)
        
    f2a = np.concatenate([var_X, lag1_acf, iact])
    f2a_cols = [f"{v}_var_X" for v in variables] + [f"{v}_lag1_acf" for v in variables] + [f"{v}_iact" for v in variables]
    return f2a, f2a_cols

def calculate_daily_lag1(day_data: pd.DataFrame, resid_cols: list[str]) -> np.ndarray:
    """Computes the lag-1 autocorrelation for a single day's residuals."""
    X_resid = day_data[resid_cols].values
    if len(X_resid) > 1:
        var_r = np.var(X_resid, axis=0)
        cov_r = np.mean((X_resid[:-1] - np.mean(X_resid, axis=0)) * (X_resid[1:] - np.mean(X_resid, axis=0)), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lag1 = np.where(var_r > 0, cov_r / var_r, 0)
    else:
        logger.debug("calculate_daily_lag1 fallback: <= 1 data point for day, returning zeros.")
        lag1 = np.zeros(len(resid_cols))
    return lag1

def compute_f2b_features(f_data: pd.DataFrame, variables: list[str]) -> tuple[np.ndarray, list[str]]:
    """Computes microscopic F2b features (mean and slope of intra-day lag-1 ACF)."""
    profile_24h = f_data.groupby('bin')[variables].mean()
    f_data = f_data.merge(profile_24h, on='bin', suffixes=('', '_mean'))
    
    resid_cols = []
    for col in variables:
        r_col = f'{col}_resid'
        f_data[r_col] = f_data[col] - f_data[f'{col}_mean']
        resid_cols.append(r_col)
        
    daily_lag1 = []
    daily_ages = []
    for age, day_data in f_data.groupby('age_days'):
        day_data = day_data.sort_values('bin')
        lag1 = calculate_daily_lag1(day_data, resid_cols)
        daily_lag1.append(lag1)
        daily_ages.append(age)
        
    daily_lag1 = np.array(daily_lag1)
    daily_ages = np.array(daily_ages)
    
    num_vars = len(variables)
    
    if len(daily_lag1) > 0:
        mean_lag1 = np.mean(daily_lag1, axis=0)
        t_center_b = daily_ages - np.mean(daily_ages)
        var_t_b = np.sum(t_center_b ** 2)
        if var_t_b > 0 and len(daily_ages) > 1:
            slope_lag1 = np.sum(t_center_b[:, None] * (daily_lag1 - mean_lag1), axis=0) / var_t_b
        else:
            logger.debug("compute_f2b_features slope fallback: var_t_b == 0 or len(daily_ages) <= 1, returning zero slope.")
            slope_lag1 = np.zeros(num_vars)
    else:
        logger.debug("compute_f2b_features fallback: no daily lag1 data, returning zeros.")
        mean_lag1 = np.zeros(num_vars)
        slope_lag1 = np.zeros(num_vars)
        
    f2b = np.concatenate([mean_lag1, slope_lag1])
    f2b_cols = [f"{v}_mean_lag1" for v in variables] + [f"{v}_slope_lag1" for v in variables]
    return f2b, f2b_cols

def dynamics(rows: pd.DataFrame, variables: list[str]) -> pd.Series:
    """
    Extracts dynamic features (F2) from a time-series of measurements.
    
    This function computes two sets of dynamic features:
    - F2a (Macroscopic): Variance, lag-1 autocorrelation, and integrated autocorrelation time (IACT)
      computed on the daily means of each variable after linear detrending.
    - F2b (Microscopic): The mean and slope (across days) of the intra-day lag-1 autocorrelation, 
      computed on residuals after subtracting the 24-hour mean profile for each bin.
      
    Args:
        rows (pd.DataFrame): DataFrame containing the time-series measurements for a single animal.
            Must include `age_days`, `bin`, and the columns listed in `variables`.
        variables (list[str]): List of column names representing the measured variables.
        
    Returns:
        pd.Series: A Series containing the concatenated F2a and F2b features for the animal, 
            with descriptive index labels.
    """
    f_data = rows.copy()
    daily_means = f_data.groupby('age_days')[variables].mean()
    
    f2a, f2a_cols = compute_f2a_features(daily_means, variables)
    f2b, f2b_cols = compute_f2b_features(f_data, variables)
    
    idx = f2a_cols + f2b_cols
    return pd.Series(np.concatenate([f2a, f2b]), index=idx)
