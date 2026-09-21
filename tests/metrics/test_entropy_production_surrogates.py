import numpy as np
import pytest
import scipy.linalg as la
from src.metrics.entropy_production import MOUResult
from src.metrics.entropy_production_surrogates import (
    reversible_gaussian_surrogate,
    phase_randomized_surrogate,
    reversible_mou_surrogate,
)
import torch

torch.autograd.set_detect_anomaly(True)

def test_phase_randomized_surrogate_prime_length():
    # ARRANGE
    rng = np.random.default_rng(42)
    N = 17  # prime number
    k = 3
    x = rng.standard_normal((N, k))
    
    # ACT
    x_surr = phase_randomized_surrogate(x, rng)
    
    # ASSERT
    assert x_surr.shape == (N, k)
    assert not np.isnan(x_surr).any()
    assert not np.isinf(x_surr).any()
    np.testing.assert_allclose(np.mean(x, axis=0), np.mean(x_surr, axis=0), atol=1e-7)
    
def test_phase_randomized_surrogate_flat_dc():
    # ARRANGE
    rng = np.random.default_rng(42)
    N = 20
    k = 3
    x = np.ones((N, k)) * 5.0  # flat DC signal
    
    # ACT
    x_surr = phase_randomized_surrogate(x, rng)
    
    # ASSERT
    assert x_surr.shape == (N, k)
    assert not np.isnan(x_surr).any()
    assert not np.isinf(x_surr).any()
    np.testing.assert_allclose(x, x_surr, atol=1e-7)

def test_phase_randomized_surrogate_nans():
    # ARRANGE
    rng = np.random.default_rng(42)
    N = 10
    k = 2
    x = rng.standard_normal((N, k))
    x[5, 0] = np.nan
    
    # ACT
    with pytest.raises(ValueError, match="Input time series contains NaNs or Infs."):
        x_surr = phase_randomized_surrogate(x, rng)

def test_reversible_gaussian_surrogate_extreme():
    # ARRANGE
    rng = np.random.default_rng(42)
    N = 19
    k = 4
    x = np.zeros((N, k)) # Exactly zero variance
    
    # ACT
    x_surr = reversible_gaussian_surrogate(x, 100.0, rng)
    
    # ASSERT
    assert x_surr.shape == (N, k)
    assert not np.isnan(x_surr).any()
    assert not np.isinf(x_surr).any()
    np.testing.assert_allclose(x, x_surr, atol=1e-7)

def test_reversible_mou_surrogate_singular():
    # ARRANGE
    rng = np.random.default_rng(42)
    k = 2
    Gamma = np.array([[1.0, 0.5], [-0.5, 1.0]])
    # singular Sigma
    Sigma = np.array([[1.0, 1.0], [1.0, 1.0]]) 
    mou = MOUResult(
        phi=0.0,
        A=np.zeros_like(Gamma),
        Gamma=Gamma,
        Q=np.zeros_like(Gamma),
        Sigma=Sigma,
        is_logm_real=True,
        is_gamma_pd=True
    )
    
    # ACT
    x_surr = reversible_mou_surrogate(mou, 100, 100.0, rng)
    
    # ASSERT
    assert x_surr.shape == (100, 2)
    assert not np.isnan(x_surr).any()
    assert not np.isinf(x_surr).any()

def test_reversible_mou_surrogate_scale_bug():
    # ARRANGE
    rng = np.random.default_rng(42)
    k = 2
    Gamma = np.array([[1.0, 0.5], [-0.5, 1.0]]) * 1e-12
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]]) * 1e-12
    mou = MOUResult(
        phi=0.0,
        A=np.zeros_like(Gamma),
        Gamma=Gamma,
        Q=np.zeros_like(Gamma),
        Sigma=Sigma,
        is_logm_real=True,
        is_gamma_pd=True
    )
    
    # ACT
    x_surr = reversible_mou_surrogate(mou, 1000, 100.0, rng)
    
    # ASSERT
    var_surr = np.var(x_surr, axis=0)
    # The variance should be close to 1e-12. If the bug exists (using 1e-8 ridge),
    # the variance will be extremely small (e.g. 1e-17).
    np.testing.assert_allclose(var_surr, np.diag(Sigma), rtol=0.5)
