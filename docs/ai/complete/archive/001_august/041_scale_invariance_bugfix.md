Context: We found a scale-invariance bug in our Waddington crash simulation.
Multiplying the noise by 0.001 preserved the eigenvalues, keeping KSM at 1.0. We
need to enforce a true thermodynamic equilibrium (absolute 0.0) and force
`metrics.py` to log its internal state to our `DiagnosticLogger`.

Task 1: Fix the Crash Physics in `src/demo/8_ephys_demo.py`

1. In `main()`, locate the Waddington Crash simulation:
   `val_seq[:, EVENT_FRAME:, :] *= 0.001`.
2. Change this to a true flatline: `val_seq[:, EVENT_FRAME:, :] = 0.0`. This
   ensures the standard deviation drops to exactly 0.0.

Task 2: Wire `ThermodynamicMetrics` to the Diagnostic Logger in
`src/metrics/metrics.py`

1. At the top of the file, ensure the logger is explicitly grabbing the
   diagnostic stream: `logger = logging.getLogger("DiagnosticLogger")`.
2. Inside `calculate_ksm`, update the threshold check to be slightly more
   forgiving of floating-point errors: `if np.std(Z_np) <= 1e-3:`.
3. Inside the sliding window loop, add a `logger.debug` statement that fires for
   _every single frame_ evaluated (not just the crash frame). It must log:
   `[PyDMD] Frame {t} | std={np.std(Z_np):.6f} | max_eig={max_eig:.4f} | KSM={ksm:.4f}`.
4. Do the same for `calculate_lle`: if `np.std(Z_np) <= 1e-3`, bypass PyDMD, set
   `max_eig = 0.0`, and set `lle = 0.0`. Add the same per-frame debug log.
