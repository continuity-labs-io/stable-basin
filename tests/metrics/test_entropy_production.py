import numpy as np
import scipy.linalg as la
import logging
import pytest

from src.metrics.entropy_production import (
    entropy_production_mou,
    entropy_production_spectral,
    entropy_production_pairwise,
)
from src.metrics.entropy_production_surrogates import (
    reversible_gaussian_surrogate,
    phase_randomized_surrogate,
    reversible_mou_surrogate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_2d_mou(q: float, n_samples: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a 2D MOU process with known exact entropy production Phi = 2 * q^2.
    A = [[-1, -q], [q, -1]], Gamma = I, Q = [[0, q], [-q, 0]], Sigma = I.
    """
    A = np.array([[-1.0, -q], [q, -1.0]])
    Sigma = np.eye(2)

    # Exact discrete transition
    M = la.expm(A * dt)
    cov_noise = Sigma - M @ Sigma @ M.T

    # Eigendecomposition for noise covariance
    vals, vecs = la.eigh(cov_noise)
    vals = np.maximum(vals, 0.0)
    L = vecs @ np.diag(np.sqrt(vals))

    x = np.zeros((n_samples, 2))
    x[0] = rng.multivariate_normal(np.zeros(2), Sigma)

    for t in range(1, n_samples):
        x[t] = M @ x[t - 1] + L @ rng.standard_normal(2)

    return x


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_mou_exact_recovery():
    # ARRANGE
    logger.info("Executing MOU closed-form validation.")
    rng = np.random.default_rng(42)
    q = 2.0
    fs = 100.0
    dt = 1.0 / fs
    n_samples = 100000
    expected_phi = 2.0 * (q**2)
    x = generate_2d_mou(q, n_samples, dt, rng)

    # ACT
    mou_res = entropy_production_mou(x, fs, lag_samples=1)
    spec_res = entropy_production_spectral(x, fs, nperseg=512)

    # ASSERT
    logger.info("Validating estimated entropy production against closed-form expectation.")
    assert mou_res.is_gamma_pd, "Gamma matrix should be positive-definite."
    np.testing.assert_allclose(mou_res.phi, expected_phi, rtol=0.1)
    assert spec_res.phi_total > 0.0, "Spectral Phi must be strictly positive."


def test_surrogates_near_zero_ep():
    # ARRANGE
    logger.info("Executing surrogate baseline validation.")
    fs = 100.0
    dt = 1.0 / fs
    n_samples = 50000
    rng = np.random.default_rng(42)

    # Generate an irreversible signal for Gaussian surrogate to squash
    x_irreversible = generate_2d_mou(q=2.0, n_samples=n_samples, dt=dt, rng=rng)
    # Generate a reversible signal for Phase-Randomized surrogate to preserve zero EP
    x_reversible = generate_2d_mou(q=0.0, n_samples=n_samples, dt=dt, rng=rng)

    # ACT
    x_rev_gauss = reversible_gaussian_surrogate(x_irreversible, fs, rng)
    x_pr = phase_randomized_surrogate(x_reversible, rng)

    mou_res_rev = entropy_production_mou(x_rev_gauss, fs, lag_samples=1)
    mou_res_pr = entropy_production_mou(x_pr, fs, lag_samples=1)

    # ASSERT
    logger.info("Validating surrogate generation functions produce near-zero entropy production.")
    assert mou_res_rev.phi < 0.5, f"Gaussian surrogate EP {mou_res_rev.phi:.3f} is not near-zero."
    assert mou_res_pr.phi < 0.5, (
        f"Phase-randomized surrogate EP {mou_res_pr.phi:.3f} is not near-zero."
    )

def test_adversarial_entropy_production():
    # ARRANGE
    import torch
    torch.autograd.set_detect_anomaly(True)
    
    fs = 100.0
    lag = 1
    
    # 1. Zero-variance signal
    x_zero = np.zeros((100, 2))
    
    # 2. NaNs
    x_nan = np.full((100, 2), np.nan)
    
    # 3. Perfectly symmetric trajectory
    x_sym = np.array([[float(i), float(i)] for i in range(50)] + [[float(49 - i), float(49 - i)] for i in range(50)])
    
    # ACT & ASSERT
    logger.info("Executing adversarial tests on entropy_production_pairwise...")
    ep_pw_zero = entropy_production_pairwise(x_zero, fs, lag)
    assert ep_pw_zero == 0.0, f"Expected 0.0 for zero variance, got {ep_pw_zero}"
    
    # NaNs should raise ValueError
    with pytest.raises(ValueError, match="NaNs or Infs"):
        entropy_production_pairwise(x_nan, fs, lag)
    
    ep_pw_sym = entropy_production_pairwise(x_sym, fs, lag)
    assert ep_pw_sym < 1e-5, f"Expected near 0.0 for symmetric trajectory, got {ep_pw_sym}"
    
    logger.info("Executing adversarial tests on entropy_production_mou...")
    ep_mou_zero = entropy_production_mou(x_zero, fs, lag)
    assert ep_mou_zero.phi == 0.0, f"Expected 0.0 for zero variance, got {ep_mou_zero.phi}"
    
    with pytest.raises(ValueError, match="NaNs or Infs"):
        entropy_production_mou(x_nan, fs, lag)
        
    ep_mou_sym = entropy_production_mou(x_sym, fs, lag)
    assert ep_mou_sym.phi < 1e-5, f"Expected near 0.0 for symmetric trajectory, got {ep_mou_sym.phi}"

    logger.info("Executing adversarial tests on entropy_production_spectral...")
    spec_zero = entropy_production_spectral(x_zero, fs, nperseg=32)
    assert spec_zero.phi_total == 0.0, "Expected 0.0 for zero variance"
    
    with pytest.raises(ValueError, match="NaNs or Infs"):
        entropy_production_spectral(x_nan, fs, nperseg=32)

