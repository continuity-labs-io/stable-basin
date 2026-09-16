Context: We found a temporal dimensionality bug in `src/metrics/metrics.py`.
When the biological input crashes to 0.0, the Mamba-2 engine outputs a constant
bias vector. This constant vector has zero variance over time, but non-zero
spatial variance across its features. Our global `np.std(Z_np)` check is missing
the flatline, allowing PyDMD to calculate an eigenvalue of 1.0 (since x*t =
x*{t+1}).

Task: Fix the flatline detection logic in `ThermodynamicMetrics`.

1. In both `calculate_ksm` and `calculate_lle`, locate the
   `Z_np = Z.T.detach().cpu().numpy()` assignment.
2. Below it, calculate the temporal standard deviation specifically:
   `temporal_std = float(np.std(Z_np, axis=1).mean())`. (This calculates the
   standard deviation across the snapshots/time for each feature, then averages
   them).
3. Replace the `if np.std(Z_np) <= 1e-3:` check with `if temporal_std <= 1e-3:`.
4. Update the debug logger to print this specific metric:
   `logger.debug(f"[PyDMD] Frame {t} | temporal_std={temporal_std:.6f} | max_eig={max_eig:.4f} | KSM={ksm:.4f}")`
   so we can watch the time-variance drop to zero.
