import numpy as np
import scipy.linalg as la
import scipy.signal as signal
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MOUResult:
    """
    Results from fitting a Multivariate Ornstein-Uhlenbeck (MOU) process.

    Attributes:
        phi: The total entropy production rate.
        A: The deterministic drift matrix.
        Gamma: The symmetric, dissipative part of A (friction matrix).
        Q: The anti-symmetric, non-equilibrium part of A (solenoidal flow).
        Sigma: The steady-state covariance matrix.
        is_logm_real: True if the matrix logarithm returned a strictly real matrix.
        is_gamma_pd: True if the resulting Gamma matrix is positive-definite.
    """

    phi: float
    A: np.ndarray
    Gamma: np.ndarray
    Q: np.ndarray
    Sigma: np.ndarray
    is_logm_real: bool
    is_gamma_pd: bool


@dataclass
class SpectralEntropyProduction:
    """
    Results from computing Spectral Entropy Production in the frequency domain.

    Attributes:
        phi_total: The total integrated spectral entropy production.
        freqs: Array of evaluated frequencies.
        phi_f: The entropy production rate at each frequency.
        S_f: The Cross-Spectral Density (CSD) matrix at each frequency.
    """

    phi_total: float
    freqs: np.ndarray
    phi_f: np.ndarray
    S_f: np.ndarray


def _check_shape(x: np.ndarray) -> np.ndarray:
    """
    Validates and centers the input time series.

    Args:
        x: Input time series array of shape (samples, channels).

    Returns:
        The mean-centered time series array.
    """
    if x.ndim != 2:
        raise ValueError(f"Input x must be 2D (samples, channels), got {x.ndim}D")
    if not np.isfinite(x).all():
        raise ValueError("Input time series contains NaNs or Infs.")
    return x - np.mean(x, axis=0)


def entropy_production_pairwise(x: np.ndarray, fs: float, lag_samples: int) -> float:
    """
    Estimator A: Lag-tau pairwise entropy production.
    Computes a lower bound on entropy production using the joint distribution of (x_t, x_{t+tau})
    and its temporal reversal.

    Args:
        x: Input time series array of shape (samples, channels).
        fs: Sampling frequency in Hz.
        lag_samples: The time lag tau expressed in number of samples.

    Returns:
        The estimated lower bound of the entropy production rate.
    """
    x = _check_shape(x)
    N, k = x.shape
    tau = lag_samples / fs

    Sigma = np.cov(x, rowvar=False)
    if k == 1:
        Sigma = np.array([[Sigma]])

    x_t = x[:-lag_samples]
    x_t_plus_tau = x[lag_samples:]

    # C(tau) = E[x_{t+tau} x_t^T]
    C = (x_t_plus_tau.T @ x_t) / (N - lag_samples)

    # Forward and backward joint covariance matrices
    S_f = np.block([[Sigma, C.T], [C, Sigma]])
    S_b = np.block([[Sigma, C], [C.T, Sigma]])

    # Add jitter to ensure invertibility
    tr_sigma = float(np.trace(Sigma))
    jitter_scale = tr_sigma if tr_sigma > 1e-12 else 1.0
    jitter = np.eye(2 * k) * 1e-8 * jitter_scale / k
    S_f += jitter
    S_b += jitter

    try:
        val = 0.5 * np.trace(np.linalg.solve(S_b, S_f)) - k
        return max(0.0, float(val / tau))
    except np.linalg.LinAlgError as e:
        logger.warning(f"LinAlgError in entropy_production_pairwise: {e}. Returning 0.0.")
        return 0.0


def entropy_production_mou(x: np.ndarray, fs: float, lag_samples: int) -> MOUResult:
    """
        Estimator B: Multivariate Ornstein-Uhlenbeck (MOU) fit.
        Estimates the continuous-time NESS matrices (A, Gamma, Q) and evaluates
        the exact analytical entropy production for the fitted linear system.

        Args:
            x: Input time series array of shape (samples, channels).
            fs: Sampling frequency in Hz.
            lag_samples: The time lag tau used to fit the MOU process.

        Returns:
    An MOUResult object containing the estimated matrices and the analytical entropy production
            rate.
    """
    x = _check_shape(x)
    N, k = x.shape
    tau = lag_samples / fs

    Sigma = np.cov(x, rowvar=False)
    if k == 1:
        Sigma = np.array([[Sigma]])

    x_t = x[:-lag_samples]
    x_t_plus_tau = x[lag_samples:]
    C = (x_t_plus_tau.T @ x_t) / (N - lag_samples)

    tr_sigma = float(np.trace(Sigma))
    ridge_scale = tr_sigma if tr_sigma > 1e-12 else 1.0
    ridge = np.eye(k) * 1e-8 * ridge_scale / k

    try:
        inv_Sigma = np.linalg.inv(Sigma + ridge)
        M = C @ inv_Sigma
        logM = la.logm(M)
    except np.linalg.LinAlgError as e:
        logger.warning(
            f"LinAlgError in entropy_production_mou during logm: {e}. Returning default MOUResult."
        )
        return MOUResult(0.0, np.zeros((k, k)), np.eye(k), np.zeros((k, k)), Sigma, False, False)

    # Diagnostic: check if the matrix logarithm is real
    is_logm_real = np.allclose(logM.imag, 0, atol=1e-5)
    A = np.real(logM) / tau

    # Decompose into dissipative (Gamma) and solenoidal (Q) components
    Gamma = -0.5 * (A @ Sigma + Sigma @ A.T)
    Q = -0.5 * (A @ Sigma - Sigma @ A.T)

    # Diagnostic: check if Gamma is positive definite
    try:
        np.linalg.cholesky(Gamma + ridge)
        is_gamma_pd = True
    except np.linalg.LinAlgError:
        is_gamma_pd = False

    try:
        inv_Gamma = np.linalg.inv(Gamma + ridge)
        phi = -np.trace(inv_Gamma @ Q @ inv_Sigma @ Q)
        phi = max(0.0, float(phi))
    except np.linalg.LinAlgError as e:
        logger.warning(f"LinAlgError in entropy_production_mou during phi calculation: {e}.")
        phi = 0.0

    return MOUResult(
        phi=phi,
        A=A,
        Gamma=Gamma,
        Q=Q,
        Sigma=Sigma,
        is_logm_real=is_logm_real,
        is_gamma_pd=is_gamma_pd,
    )


def entropy_production_spectral(
    x: np.ndarray, fs: float, nperseg: int, fmax: Optional[float] = None
) -> SpectralEntropyProduction:
    """
        Estimator C: Exact Gaussian Spectral Entropy Production.
        Computes the cross-spectral density matrix S(f) via Welch's method and integrates
        the irreversibility density phi(f) over frequencies.

        Args:
            x: Input time series array of shape (samples, channels).
            fs: Sampling frequency in Hz.
            nperseg: Length of each segment for Welch's method.
    fmax: Optional maximum frequency to include in the integration. If None, integrates up to
            Nyquist.

        Returns:
    A SpectralEntropyProduction object containing the total integrated entropy production and
            frequency-resolved data.
    """
    x = _check_shape(x)
    N, k = x.shape

    step = nperseg // 2
    n_segments = (N - nperseg) // step + 1

    if n_segments < 1:
        raise ValueError(f"nperseg ({nperseg}) is too large for data length ({N})")

    # Vectorized Welch's Method for Cross-Spectral Matrix
    window = signal.windows.hann(nperseg)
    win_norm = 1.0 / (fs * (window**2).sum())

    # Segment the data [n_segments, nperseg, k]
    segments = np.array([x[i * step : i * step + nperseg] for i in range(n_segments)])
    segments = segments * window[:, np.newaxis]

    # FFT to frequency domain
    X_f = np.fft.rfft(segments, axis=1)  # [n_segments, n_freqs, k]
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    n_freqs = len(freqs)

    # Compute Cross-Spectrum S(f) efficiently via einsum: E[X_a X_b^*]
    S_f = np.einsum("sfi,sfj->fij", X_f, X_f.conj()) * (win_norm / n_segments)

    phi_f = np.zeros(n_freqs)

    # Calculate a dynamic ridge based on the average power across channels and frequencies
    avg_power = np.trace(np.mean(S_f, axis=0)).real
    jitter = np.eye(k) * 1e-10 * (avg_power / k if avg_power > 0 else 1.0)

    for i in range(n_freqs):
        Sf = S_f[i] + jitter

        try:
            # phi(f) = Tr(S(f)^{-T} S(f)) - k
            inv_Sf_T = np.linalg.inv(Sf.T)
            val = np.real(np.trace(inv_Sf_T @ Sf)) - k
            phi_f[i] = max(0.0, val)
        except np.linalg.LinAlgError as e:
            logger.warning(
                f"LinAlgError in entropy_production_spectral at frequency index {i}: {e}."
            )
            phi_f[i] = 0.0

    if fmax is not None:
        mask = freqs <= fmax
        freqs = freqs[mask]
        phi_f = phi_f[mask]
        S_f = S_f[mask]

    # Multiply by 2.0 to account for negative frequencies in the integral
    from scipy.integrate import trapezoid

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    phi_total = 2.0 * float(trapezoid(phi_f, dx=df))

    return SpectralEntropyProduction(phi_total=phi_total, freqs=freqs, phi_f=phi_f, S_f=S_f)


def band_entropy_production(spec: SpectralEntropyProduction, band: Tuple[float, float]) -> float:
    """
    Computes band-resolved Entropy Production by integrating phi(f) over a specific range.

    Args:
        spec: The SpectralEntropyProduction object containing frequency-resolved phi(f).
        band: A tuple (f_min, f_max) defining the frequency band to integrate over.

    Returns:
        The integrated entropy production within the specified frequency band.
    """
    f_min, f_max = band
    mask = (spec.freqs >= f_min) & (spec.freqs <= f_max)

    if not np.any(mask):
        return 0.0

    from scipy.integrate import trapezoid

    df = spec.freqs[1] - spec.freqs[0] if len(spec.freqs) > 1 else 1.0
    return 2.0 * float(trapezoid(spec.phi_f[mask], dx=df))
