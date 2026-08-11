# TASK 4: Build the Human Digital Twin Simulation Demo

We need a capstone demo script to simulate a full clinical rejuvenation pass on a 50-year-old human. We will use `MeldEngine` as a "Digital Twin", apply our `RejuvenationFlightController` to guide the therapy, and plot the resulting telemetry.

Please create a new file: `src/demo/10_human_rejuvenation_sim.py`

**Requirements:**
1. **Setup:** Instantiate the `HumanTelemetryLoader`, `EpigeneticEntropyLoader`, `MeldEngine`, `ThermodynamicMetrics`, and `RejuvenationFlightController`. Force device to CPU for the simulation to avoid MPS autograd issues.
2. **The Simulation Loop (The Ratchet Mechanism):** Simulate a 20-minute timeline.
    *   **Phase 1 (Baseline, Min 0-5):** Run the engine on healthy baseline data to establish the patient's deep `epsilon_0` (KSM) stability well.
    *   **Phase 2 (The Shock, Min 5-10):** Apply the `apply_therapy_shock()` from the dataloader. The `RejuvenationFlightController` must process this, detect the KSM drop / CSD spike, and fire the `EMERGENCY_ABORT`.
    *   **Phase 3 (Recovery, Min 10-15):** The abort clears the shock. Telemetry recovers. The controller resumes infusion.
    *   **Phase 4 (The Delta, Min 15-20):** Call the `EpigeneticEntropyLoader` with `biological_age=45`. Calculate the configurational entropy dispersion.
3. **Visualization Dashboard:** Use `matplotlib` (using the `Agg` backend) to generate a 4-panel dark-mode dashboard saved to `output/10_human_rejuvenation_sim.png`:
    *   **Panel 1:** The Raw Wearable Data (HRV trace) showing the shock and recovery. Overlay a background color (Green = IV ON, Red = IV ABORTED).
    *   **Panel 2:** The KSM ($\epsilon_0$) trajectory. Highlight the 0.85 threshold line. Mark the exact moment the controller paused the infusion with a vertical red line.
    *   **Panel 3:** The CSD (Variance / Wobble) showing the biological noise ($D_0$) spiking during the shock.
    *   **Panel 4:** The Epigenetic Variance ($Z$) plotting a step-function reduction from Age 50 to Age 45, proving we successfully scooped configurational entropy out of the system without killing the patient.
