"""
synthetic_aging.py  ->  suggested location: src/data/behavior/synthetic_aging.py

Known-ground-truth degradations for validating the Stable Basin pipeline.
Nothing here models biological aging. Each function injects a specific, labelled
change so a detector can be scored against the truth.

Tier 1 (pure synthetic, analytic ground truth)
    ou_process(n, dt, tau, sigma)      Ornstein-Uhlenbeck, exact discretisation.
                                       tau and/or sigma may be arrays (ramps).
        tau ramp,  sigma fixed  -> critical slowing down: AR1 and variance both rise.
        sigma ramp, tau fixed   -> NULL look-alike: variance rises, AR1 flat.
    add_measurement_noise(x, sd)       NULL look-alike: variance rises, AR1 falls.
                                       (This is what the old `is_aged` transform did.)

Tier 2 (real gait, slowed restoring dynamics)
    slow_amplitude_relaxation(traj, slowdown, pair)
        Treats one eigenworm pair as the gait oscillator, splits it into amplitude
        and phase, and slows the relaxation of log-amplitude deviations by a factor
        `slowdown` while keeping the real innovations and the real phase. At
        slowdown == 1 it returns the input unchanged (up to float error).
        In the linearised Langevin picture (tau ~ 1/lambda, var ~ T/lambda), a
        slowdown of s at fixed innovation variance corresponds to lambda -> lambda/s
        along the amplitude direction.

CSD indicators
    rolling_indicators(x, window)      rolling variance and lag-1 AR1 on detrended windows
    kendall_trend(series)              Kendall tau of an indicator against time
    amplitude_residual_stats(traj)     AR1 / variance of log-amplitude deviations
                                       (the manipulation check for Tier 2)
Compute CSD indicators on residuals about the cycle, not on raw eigenworm values:
raw lag-1 AR1 of a ~0.5 Hz undulation sampled at 16-25 Hz sits near cos(2*pi*f*dt)
~ 0.99 regardless of resilience, so it mostly measures waveform smoothness.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

ArrayLike = float | np.ndarray


def _as_series(v: ArrayLike, n: int) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        return np.full(n, float(v))
    if v.shape[0] != n:
        raise ValueError(f"Expected scalar or length-{n} array, got shape {v.shape}.")
    return v


# --------------------------------------------------------------------------- Tier 1

def ou_process(
    n: int,
    dt: float,
    tau: ArrayLike,
    sigma: ArrayLike = 1.0,
    x0: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """dx = -(x / tau) dt + sigma dW, exact per-step discretisation.

    Stationary AR1 at lag dt is exp(-dt / tau); stationary variance is sigma^2 tau / 2.
    """
    rng = rng or np.random.default_rng()
    tau_t = _as_series(tau, n)
    sig_t = _as_series(sigma, n)
    if np.any(tau_t <= 0):
        raise ValueError("tau must be positive.")
    a = np.exp(-dt / tau_t)
    step_sd = sig_t * np.sqrt(0.5 * tau_t * (1.0 - a**2))
    x = np.empty(n)
    x[0] = rng.normal(0.0, sig_t[0] * np.sqrt(0.5 * tau_t[0])) if x0 is None else x0
    xi = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = a[t] * x[t - 1] + step_sd[t] * xi[t]
    return x


def add_measurement_noise(
    x: np.ndarray, sd: ArrayLike, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Additive white observation noise; sd may ramp over time (axis 0)."""
    rng = rng or np.random.default_rng()
    sd_t = _as_series(sd, x.shape[0])
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x + sd_t.reshape(shape) * rng.standard_normal(x.shape)


# --------------------------------------------------------------------------- Tier 2

def phase_amplitude(traj: np.ndarray, pair: tuple[int, int] = (0, 1)):
    """Amplitude and phase of the oscillator spanned by two eigenworm channels.

    Check on real data that this pair traces a ring during forward crawling
    before trusting it as the gait oscillator.
    """
    a1, a2 = traj[:, pair[0]], traj[:, pair[1]]
    return np.hypot(a1, a2), np.arctan2(a2, a1)


def _log_amp_deviation(r: np.ndarray):
    eps = 1e-3 * max(float(np.median(r)), 1e-12)
    u = np.log(r + eps)
    c = float(np.mean(u))
    return u - c, c, eps


def _ar1_coef(d: np.ndarray) -> float:
    den = float(np.dot(d[:-1], d[:-1]))
    if den <= 0:
        return 0.0
    return float(np.clip(np.dot(d[1:], d[:-1]) / den, 0.0, 0.9999))


def slow_amplitude_relaxation(
    traj: np.ndarray,
    slowdown: ArrayLike,
    pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Slow the relaxation of gait-amplitude deviations by `slowdown` (scalar or per-frame).

    The log-amplitude deviation d_t is written as d_t = a d_{t-1} + e_t, with a fitted
    at lag 1 and e_t the real innovations. The pole is moved to a^(1/s), i.e. the
    relaxation time is multiplied by s, and d is re-integrated from the same e_t.
    Phase and all other channels are untouched.
    """
    traj = np.asarray(traj, dtype=float)
    n = traj.shape[0]
    s_t = _as_series(slowdown, n)
    if np.any(s_t <= 0):
        raise ValueError("slowdown must be positive.")

    r, phi = phase_amplitude(traj, pair)
    d, c, eps = _log_amp_deviation(r)
    a = _ar1_coef(d)

    e = np.empty(n)
    e[0] = d[0]
    e[1:] = d[1:] - a * d[:-1]

    a_new = a ** (1.0 / s_t)
    d_new = np.empty(n)
    d_new[0] = d[0]
    for t in range(1, n):
        d_new[t] = a_new[t] * d_new[t - 1] + e[t]

    r_new = np.exp(d_new + c) - eps
    out = traj.copy()
    out[:, pair[0]] = r_new * np.cos(phi)
    out[:, pair[1]] = r_new * np.sin(phi)
    return out


# --------------------------------------------------------------------------- indicators

def amplitude_residual_stats(traj: np.ndarray, pair: tuple[int, int] = (0, 1)) -> dict:
    """Manipulation check: AR1 and variance of log-amplitude deviations."""
    r, _ = phase_amplitude(traj, pair)
    d, _, _ = _log_amp_deviation(r)
    return {"amp_ar1": _ar1_coef(d), "amp_var": float(np.var(d))}


def rolling_indicators(x: np.ndarray, window: int, detrend: bool = True) -> dict:
    """Rolling variance and lag-1 autocorrelation of a 1-D series.

    Each window is linearly detrended first. Report the two indicators separately;
    summing them mixes units and makes the score depend on signal scale.
    """
    x = np.asarray(x, dtype=float)
    if x.shape[0] < window or window < 3:
        raise ValueError("Series shorter than window, or window < 3.")
    w = np.lib.stride_tricks.sliding_window_view(x, window)
    resid = w - w.mean(axis=1, keepdims=True)
    if detrend:
        tc = np.arange(window, dtype=float) - (window - 1) / 2.0
        slope = resid @ tc / float(tc @ tc)
        resid = resid - slope[:, None] * tc[None, :]
    var = resid.var(axis=1)
    den = np.einsum("ij,ij->i", resid[:, :-1], resid[:, :-1])
    num = np.einsum("ij,ij->i", resid[:, 1:], resid[:, :-1])
    ar1 = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return {"var": var, "ar1": ar1}


def kendall_trend(series: np.ndarray) -> float:
    """Kendall tau of an indicator against time. Judge significance against
    surrogates (phase-randomised or fitted-AR(1)), not the analytic p-value,
    because rolling-window indicators are heavily autocorrelated."""
    tau, _ = kendalltau(np.arange(len(series)), series)
    return float(tau)


# --------------------------------------------------------------------------- self-test

def _stuart_landau(n: int, dt: float, freq: float, mu: float, noise: float, rng) -> np.ndarray:
    """Noisy limit-cycle oscillator; radial relaxation rate ~ 2*mu."""
    z = np.empty(n, dtype=complex)
    z[0] = np.sqrt(mu)
    omega = 2 * np.pi * freq
    sq = np.sqrt(dt) * noise
    for t in range(1, n):
        zt = z[t - 1]
        dz = ((mu + 1j * omega) * zt - (abs(zt) ** 2) * zt) * dt
        z[t] = zt + dz + sq * (rng.standard_normal() + 1j * rng.standard_normal())
    return np.stack([z.real, z.imag], axis=1)


def _self_test(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fs, n, win = 25.0, 20000, 500
    dt = 1.0 / fs

    n_seeds = 30
    print(f"Tier 1: Kendall tau of rolling indicators over {n_seeds} seeds, mean [5th, 95th pct]")
    makers = {
        "CSD      tau 0.5->2.0 s, sigma fixed": lambda g: ou_process(n, dt, np.linspace(0.5, 2.0, n), 1.0, rng=g),
        "NULL-a   sigma x1->x2, tau fixed   ": lambda g: ou_process(n, dt, 0.5, np.linspace(1.0, 2.0, n), rng=g),
        "NULL-b   meas. noise sd 0->0.5     ": lambda g: add_measurement_noise(
            ou_process(n, dt, 0.5, 1.0, rng=g), np.linspace(0.0, 0.5, n), rng=g
        ),
        "NULL-c   stationary (no change)    ": lambda g: ou_process(n, dt, 0.5, 1.0, rng=g),
    }
    for name, make in makers.items():
        kv, ka = [], []
        for k in range(n_seeds):
            ind = rolling_indicators(make(np.random.default_rng(1000 + k)), win)
            kv.append(kendall_trend(ind["var"]))
            ka.append(kendall_trend(ind["ar1"]))
        fmt = lambda v: f"{np.mean(v):+.2f} [{np.percentile(v, 5):+.2f}, {np.percentile(v, 95):+.2f}]"
        print(f"  {name}  var: {fmt(kv)}   AR1: {fmt(ka)}")

    print("\nTier 2: amplitude slowing on a noisy 0.5 Hz limit cycle sampled at 25 Hz")
    osc = _stuart_landau(n, dt, freq=0.5, mu=1.0, noise=0.3, rng=rng)
    traj = np.concatenate([osc, 0.3 * rng.standard_normal((n, 4))], axis=1)
    same = slow_amplitude_relaxation(traj, 1.0)
    print(f"  identity check (s=1) max |diff| = {np.max(np.abs(same - traj)):.2e}")
    print("  s     raw lag-1 AR1 (ch0)   amp-residual AR1   amp-residual var   peak freq (Hz)")
    for s in (1.0, 1.5, 2.0, 3.0):
        y = slow_amplitude_relaxation(traj, s)
        c0 = y[:, 0] - y[:, 0].mean()
        raw_ar1 = float(np.dot(c0[1:], c0[:-1]) / np.dot(c0[:-1], c0[:-1]))
        st = amplitude_residual_stats(y)
        spec = np.abs(np.fft.rfft(c0)) ** 2
        f = np.fft.rfftfreq(n, dt)
        print(f"  {s:<5} {raw_ar1:>12.4f} {st['amp_ar1']:>20.4f} {st['amp_var']:>18.4f} {f[np.argmax(spec[1:]) + 1]:>16.3f}")

    print("\n  Within-recording ramp s: 1 -> 3 (detection-latency test)")
    y = slow_amplitude_relaxation(traj, np.linspace(1.0, 3.0, n))
    d, _, _ = _log_amp_deviation(phase_amplitude(y)[0])
    ind = rolling_indicators(d, win)
    raw = rolling_indicators(y[:, 0], win)
    print(f"  amp-residual  var: {kendall_trend(ind['var']):+.2f}   AR1: {kendall_trend(ind['ar1']):+.2f}")
    print(f"  raw channel 0 var: {kendall_trend(raw['var']):+.2f}   AR1: {kendall_trend(raw['ar1']):+.2f}")


if __name__ == "__main__":
    _self_test()
