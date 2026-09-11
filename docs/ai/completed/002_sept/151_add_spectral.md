Load src/system/telemetry_logger.py and src/demo/05_flight_recorder_demo.py.

In telemetry_logger.py, add a new method log_spectral_decoherence(self, plv_coherence: float, cfc_enslavement: float) that logs these two scalars to the Rerun viewer under the path "early_warning_radar/spectral".

In 05_flight_recorder_demo.py, import SpectralMetrics. Inside the simulation loop, calculate the PLV across the latent dimensions of z_pert_seq, and calculate the CFC between a designated "slow" dimension and "fast" dimension of the latent state.

Stream these computed metrics out using the new exhaust.log_spectral_decoherence method.

Maintain a calm, professional internal tone throughout the modifications.
