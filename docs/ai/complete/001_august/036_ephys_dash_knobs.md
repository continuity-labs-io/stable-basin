Context: We need to polish the `8_ephys_demo.py` dashboard. The loss is
currently exploding into the millions, the PyDMD KSM metric is plotting random
noise instead of a sharp collapse, and the script lacks easily adjustable
parameters.

Task: Refactor `src/demo/8_ephys_demo.py` to expose presentation knobs and fix
the physics constraints.

1. Add a clear "DEMO KNOBS" section at the top of `main()`. Expose:
   - `BURN_IN_ITERATIONS` (default 10)
   - `SEQUENCE_LENGTH_MS` (default 500)
   - `CRASH_INJECTION_MS` (default 250)
   - `SAMPLING_RATE_HZ` (default 20000)
   - Convert these milliseconds into actual frame indices (e.g.,
     `SEQ_LEN = int(SEQUENCE_LENGTH_MS * SAMPLING_RATE_HZ / 1000)`).
2. Because the new `brw_dataloader.py` now provides Z-scored data, the
   `MeldLoss` calculation will naturally output small, stable numbers. Log
   these.
3. Fix the "Waddington Crash" simulation. Instead of adding random noise at
   `EVENT_FRAME` (which PyDMD treats as high-rank stability), simulate a true
   biological "flatline": force all tensor values after `EVENT_FRAME` to exactly
   `0.0` (or add a massive static DC offset). This will force the Dynamic Mode
   Decomposition eigenvalues to collapse, causing KSM to plummet as intended.
4. Update `plot_ephys_dashboard` to use the exposed `CRASH_INJECTION_MS` for its
   vertical line labels, ensuring the dashboard titles accurately reflect the
   tuned parameters.
