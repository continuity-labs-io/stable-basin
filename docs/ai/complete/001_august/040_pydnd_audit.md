Context: We are diagnosing the PyDMD KSM metric in `src/metrics/metrics.py`.
When we feed it a flatlined tensor (simulating a biological crash where variance
drops near zero), the KSM score is bouncing back to 1.0 instead of collapsing to
0.0. This implies the solver is silently failing on rank-deficient matrices and
triggering our graceful fallback.

Task: Update `calculate_ksm` in `ThermodynamicMetrics` to audit the PyDMD
solver.

1. Add an explicit variance check at the top of the sliding window loop. If the
   standard deviation of `Z_np` is less than `1e-4`, bypass PyDMD entirely,
   force `max_eig = 0.0` (which will yield KSM ~0.60, or adjust KSM calculation
   to force 0.0), and log an explicit DEBUG message: "Flatline detected at frame
   {t}, forcing rank collapse."
2. Remove the `warnings.simplefilter("ignore")` block around the PyDMD fit. We
   want to see the exact SVD or LinAlg warnings.
3. In the `except Exception as e:` block, use the `logging` module to log the
   exact exception message (e.g.,
   `logger.error(f"PyDMD Failed at frame {t}: {e}")`).
4. Instead of falling back to `max_eig = 1.0` (which artificially tells the
   dashboard the system is perfectly stable), change the fallback to
   `max_eig = 0.0` so any mathematical failure correctly registers as a drop in
   thermodynamic stability on the graph.
