import numpy as np
import scipy.linalg as la
from src.metrics.entropy_production import MOUResult, _check_shape


def reversible_gaussian_surrogate(x: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Primary Null: Reversible Gaussian Surrogate.
    Synthesizes a time series with the exact same power spectra and *real* coherence
    as the input, but forces the imaginary cross-spectra (phase lags) to exactly zero.
    Produces zero Entropy Production by construction.

    Args:
        x: Input time series array of shape (samples, channels).
        fs: Sampling frequency in Hz.
        rng: NumPy random generator instance.

    Returns:
        The surrogate time series array with the same shape as the input.
    """
    x_c = _check_shape(x)
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
    (and thus the Gaussian entropy production) while destroying non-Gaussian time-domain features.

    Args:
        x: Input time series array of shape (samples, channels).
        rng: NumPy random generator instance.

    Returns:
        The phase-randomized surrogate time series array with the same shape as the input.
    """
    x_c = _check_shape(x)
    N, k = x_c.shape

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
    Uses the matrices fitted by Estimator B (entropy_production_mou) but forces the solenoidal
    flow (Q) to zero. Generates a time series through exact discrete unrolling
    to avoid Euler-Maruyama discretization errors.

    Args:
        mou: An MOUResult object containing the fitted process matrices.
        n_samples: Number of samples to generate.
        fs: Sampling frequency in Hz.
        rng: NumPy random generator instance.

    Returns:
        The generated reversible surrogate time series array of shape (n_samples, channels).
    """
    k = mou.Gamma.shape[0]
    dt = 1.0 / fs

    # Reversible Drift Matrix: A_rev = -Gamma @ Sigma^-1
    tr_sigma = float(np.trace(mou.Sigma))
    ridge_scale = tr_sigma if tr_sigma > 1e-12 else 1.0
    ridge = np.eye(k) * 1e-8 * ridge_scale / k
    inv_Sigma = np.linalg.inv(mou.Sigma + ridge)
    
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
