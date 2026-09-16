Load src/core/rejuvenation_controller.py.

Update the RejuvenationFlightController to utilize the new frequency domain metrics.

In process_telemetry_chunk, instantiate SpectralMetrics and compute the Phase-Locking Value (PLV) across the spatial dimensions of the latent sequence (z_seq).

Add a CRITICAL_PLV_THRESHOLD (e.g., 0.65) to the __init__ method.

Update evaluate_safety_margins to trigger an "EMERGENCY_ABORT" or "WARNING" if the plv_score drops below this threshold, indicating spectral decoherence.

Ensure all logging outputs remain calm, subdued, and professional.
