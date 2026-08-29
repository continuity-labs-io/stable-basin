# Inertial Dial Validation Suite

The Inertial Dial Validation Suite: In
`tests/models/ssm/physics/test_inertial_routing.py`, build an automated
telemetry testing suite using pytest and matplotlib.

1. **Generate a synthetic biological dataset**: Reuse the existing
   `GeviInjector` from `src.data.sim2real.gevi_injector`. Generate a baseline
   signal with a high-variance anomaly phase (e.g. `is_healthy=False`).
2. **Implement a simplified mock of the Conv1D kinematic routing**: Implement
   `MockKinematicRouting(nn.Module)`. Use `kernel_size=4` acting as a finite
   difference estimator (e.g. `[-1, 3, -3, 1]`) for acceleration/jerk, and the
   $\Delta t$ gating mechanism (`softplus(abs(dx))`).
3. **Write an assertion and output a 2-panel Matplotlib graph**
   (`output/tests/models/ssm/physics/inertia_dial_validation.png`) demonstrating
   that the $\Delta t$ "Inertial Dial" correctly isolates the signals.
   - The top panel should show the raw biological signal.
   - The bottom panel must show low $\Delta t$ (high inertia / state
     preservation) during the slow baseline drift, and a violent spike in
     $\Delta t$ (low inertia / aggressive state rewrite) exactly when the
     high-frequency anomalies occur.
