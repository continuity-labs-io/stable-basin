# TASK 3: Build the Closed-Loop Rejuvenation Controller

We cannot blindly administer Level 3 rejuvenation therapies. The biological noise ($D_0$) will overwhelm the patient's intrinsic resilience ($\epsilon_0$). We must build a controller that reads the MELD engine's real-time thermodynamic metrics (CSD and KSM) and physically controls the payload delivery to avoid a saddle-node bifurcation.

Please create a new file: `src/core/rejuvenation_controller.py`

**Requirements:**
1. **Class Name:** `RejuvenationFlightController`
2. **Dependencies:** It takes an instance of `MeldEngine` and `ThermodynamicMetrics`. Implement a method `process_telemetry_chunk(x_raw, mask)` that pushes data through the engine and returns current KSM and CSD.
3. **The Control Logic (PID / State Machine):** Implement `evaluate_safety_margins(ksm_score, csd_score)`.
    *   Define safety thresholds: `CRITICAL_KSM_THRESHOLD = 0.85` and `MAX_CSD_VARIANCE = 3.0` (relative to baseline).
    *   **STATE_NOMINAL:** If `ksm_score > 0.92`, return `{"action": "MAINTAIN_INFUSION", "status": "SAFE"}`.
    *   **STATE_BIFURCATION_DANGER:** If `ksm_score < CRITICAL_KSM_THRESHOLD` or `csd_score > MAX_CSD_VARIANCE`, the patient is undergoing *Critical Slowing Down* (approaching systemic collapse). Return `{"action": "EMERGENCY_ABORT", "status": "CRITICAL", "reason": "Saddle-node bifurcation imminent."}`.
4. **Hardware Webhook:** Implement a mock method `_actuate_iv_pump(action)` that logs a highly visible warning/update to the console using the standard `logging` module.
5. **Hysteresis:** Implement a basic dampening function (e.g., KSM must be below threshold for 3 consecutive frames) so the IV pump doesn't rapidly toggle on and off every millisecond due to micro-fluctuations in the latent state.
