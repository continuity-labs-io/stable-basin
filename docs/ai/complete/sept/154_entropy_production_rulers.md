# Phase 1, Prompt 1: The Entropy Production Rulers

Please create the following two files in `src/metrics/` to implement the linear estimators and surrogate nulls. These rely strictly on `numpy` and `scipy` to establish our mathematical ground truth before touching deep learning.

**File 1: `src/metrics/entropy_production.py`**

```python
import numpy as np
import scipy.linalg as la
import scipy.signal as signal
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional


@dataclass
class MOUResult:
    phi: float
    A: np.ndarray
    Gamma: np.ndarray
    Q: np.ndarray
    Sigma: np.ndarray
    is_logm_real: bool
    is_gamma_pd: bool


@dataclass
class SpectralEP:
    phi_total: float
    freqs: np.ndarray
    phi_f: np.ndarray
    S_f: np.ndarray


def _check_shape(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Input x must be 2D (samples, channels), got {x.ndim}D")
    return x - np.mean(x, axis=0)


def ep_pairwise(x: np.ndarray, fs: float, lag_samples: int) -> float:
    """
    Estimator A: Lag-tau pairwise entropy production.
    Computes a lower bound on EP using the joint distribution of (x_t, x_{t+tau})
    and its temporal reversal.
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
    jitter = np.eye(2 * k) * 1e-8 * np.trace(Sigma) / k
    S_f += jitter
    S_b += jitter

    try:
        val = 0.5 * np.trace(np.linalg.solve(S_b, S_f)) - k
        return max(0.0, float(val / tau))
    except np.linalg.LinAlgError:
        return 0.0


def ep_mou(x: np.ndarray, fs: float, lag_samples: int) -> MOUResult:
    """
    Estimator B: Multivariate Ornstein-Uhlenbeck (MOU) fit.
    Estimates the continuous-time NESS matrices (A, Gamma, Q) and evaluates
    the exact analytical EP for the fitted linear system.
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

    ridge = np.eye(k) * 1e-8 * np.trace(Sigma) / k

    try:
        inv_Sigma = np.linalg.inv(Sigma + ridge)
        M = C @ inv_Sigma
        logM = la.logm(M)
    except np.linalg.LinAlgError:
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
    except np.linalg.LinAlgError:
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


def ep_spectral(x: np.ndarray, fs: float, nperseg: int, fmax: Optional[float] = None) -> SpectralEP:
    """
    Estimator C: Exact Gaussian Spectral Entropy Production.
    Computes the cross-spectral density matrix S(f) via Welch's method and integrates
    the irreversibility density phi(f) over frequencies.
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
        except np.linalg.LinAlgError:
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

    return SpectralEP(phi_total=phi_total, freqs=freqs, phi_f=phi_f, S_f=S_f)


def band_ep(spec: SpectralEP, band: Tuple[float, float]) -> float:
    """
    Computes band-resolved Entropy Production by integrating phi(f) over a specific range.
    """
    f_min, f_max = band
    mask = (spec.freqs >= f_min) & (spec.freqs <= f_max)

    if not np.any(mask):
        return 0.0

    from scipy.integrate import trapezoid

    df = spec.freqs[1] - spec.freqs[0] if len(spec.freqs) > 1 else 1.0
    return 2.0 * float(trapezoid(spec.phi_f[mask], dx=df))
```

**File 2: `src/metrics/ep_surrogates.py`**

```python
import numpy as np
import scipy.linalg as la
from src.metrics.entropy_production import MOUResult


def reversible_gaussian_surrogate(x: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Primary Null: Reversible Gaussian Surrogate.
    Synthesizes a time series with the exact same power spectra and *real* coherence
    as the input, but forces the imaginary cross-spectra (phase lags) to exactly zero.
    Produces zero Entropy Production by construction.
    """
    x_c = x - np.mean(x, axis=0)
    N, k = x_c.shape

    # Take full-sequence FFT to preserve all finite-length spectral properties without windowing
    X = np.fft.rfft(x_c, axis=0)  # [n_freqs, k]
    n_freqs = X.shape[0]

    # Compute empirical rank-1 cross-spectrum and explicitly strip imaginary phase lags
    S_f = X[:, :, np.newaxis] * X[:, np.newaxis, :].conj()
    Re_S_f = np.real(S_f)

    Z = np.zeros_like(X)

    for i in range(n_freqs):
        # Eigendecomposition stably factors the positive semi-definite matrix.
        vals, vecs = np.linalg.eigh(Re_S_f[i])
        vals = np.maximum(vals, 0.0)

        # L = V @ sqrt(Lambda)
        L = vecs @ np.diag(np.sqrt(vals))

        # Draw independent complex Gaussian noise
        if i == 0 or (N % 2 == 0 and i == n_freqs - 1):
            # DC and Nyquist components must be purely real
            noise = rng.standard_normal(k)
        else:
            noise = (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / np.sqrt(2.0)

        Z[i] = L @ noise

    # Inverse FFT back to time domain
    x_surr = np.fft.irfft(Z, n=N, axis=0)

    # Rescale to perfectly match original variance
    std_orig = np.std(x, axis=0)
    std_surr = np.std(x_surr, axis=0) + 1e-10

    return x_surr * (std_orig / std_surr) + np.mean(x, axis=0)


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sensitivity Null: Multivariate Phase Randomization.
    Adds a random phase shift at each frequency, but crucially applies the SAME
    shift to all channels. This exactly preserves the full complex cross-spectrum S(f)
    (and thus the Gaussian EP) while destroying non-Gaussian time-domain features.
    """
    N, k = x.shape
    x_c = x - np.mean(x, axis=0)

    X = np.fft.rfft(x_c, axis=0)
    n_freqs = X.shape[0]

    # Generate random phase shifts [0, 2pi)
    phases = rng.uniform(0, 2 * np.pi, size=(n_freqs, 1))

    # DC component must remain real
    phases[0] = 0.0
    if N % 2 == 0:
        phases[-1] = 0.0  # Nyquist must remain real

    # Apply the same phase shift to all channels
    X_surr = X * np.exp(1j * phases)

    # Inverse FFT back to time domain
    x_surr = np.fft.irfft(X_surr, n=N, axis=0)

    return x_surr + np.mean(x, axis=0)


def reversible_mou_surrogate(
    mou: MOUResult, n_samples: int, fs: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Model-based Null: Reversible MOU Process.
    Uses the matrices fitted by Estimator B (ep_mou) but forces the solenoidal
    flow (Q) to zero. Generates a time series through exact discrete unrolling
    to avoid Euler-Maruyama discretization errors.
    """
    k = mou.Gamma.shape[0]
    dt = 1.0 / fs

    # Reversible Drift Matrix: A_rev = -Gamma @ Sigma^-1
    inv_Sigma = np.linalg.inv(mou.Sigma + np.eye(k) * 1e-8)
    A_rev = -mou.Gamma @ inv_Sigma

    # Exact discrete state transition matrix: M = exp(A_rev * dt)
    M = la.expm(A_rev * dt)

    # Discrete noise covariance: \Sigma - M \Sigma M^T
    cov_noise = mou.Sigma - M @ mou.Sigma @ M.T

    # Eigen-decomposition for stable generation
    vals, vecs = np.linalg.eigh(cov_noise)
    vals = np.maximum(vals, 0.0)
    L = vecs @ np.diag(np.sqrt(vals))

    x_surr = np.zeros((n_samples, k))

    # Draw initial state from stationary distribution
    x_surr[0] = rng.multivariate_normal(np.zeros(k), mou.Sigma)

    # Exact discrete unrolling
    for t in range(1, n_samples):
        noise = rng.standard_normal(k)
        x_surr[t] = M @ x_surr[t - 1] + L @ noise

    return x_surr
```
