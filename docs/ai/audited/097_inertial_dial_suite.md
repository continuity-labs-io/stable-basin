The Inertial Dial Validation Suite: In tests/models/ssm/physics/test_inertial_routing.py, build an automated telemetry testing suite using pytest and matplotlib.

Generate a synthetic biological dataset: A slow, continuous baseline sine wave (representing a healthy macroscopic rhythm) superimposed with sharp, high-frequency square spikes (simulating rapid phase transitions or valid ion-channel events).

Implement a simplified mock of the Conv1D kinematic routing (kernel size 4 acting as a finite difference estimator for acceleration/jerk) and the Δt gating mechanism.

Write an assertion and output a 2-panel Matplotlib graph (tests/plots/inertia_dial_validation.png) demonstrating that the Δt "Inertial Dial" correctly isolates the signals. The top panel should show the raw biological signal. The bottom panel must show low Δt (high inertia / state preservation) during the slow baseline drift, and a violent spike in Δt (low inertia / aggressive state rewrite) exactly when the high-frequency anomalies occur.
