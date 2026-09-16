Load src/metrics/time_domain.py and src/metrics/spectral.py.

Update the ThermodynamicMetrics class in time_domain.py to serve as a unified diagnostic engine that seamlessly bridges both time and frequency domains.

1. Update the __init__ method of ThermodynamicMetrics to instantiate and store a SpectralMetrics object (e.g., self.spectral_metrics = SpectralMetrics()).
2. Add a new method, calculate_unified_diagnostics(self, z_seq, raw_telemetry, macro_channel_idx, micro_channel_idx, sampling_rate), to the class.
3. In this new method, call the existing time-domain methods (calculate_ksm, calculate_csd) on z_seq.
4. Also within this method, use the stored self.spectral_metrics to call the frequency-domain methods: calculate_psd (on the telemetry or latent state) and calculate_cfc_pac (between the specified macro and micro channels).
5. Return all computed metrics in a single, structured dictionary (e.g., {"time_domain": {"ksm": ..., "csd": ...}, "frequency_domain": {"psd": ..., "pac": ...}}).

This provides a single, cohesive interface for higher-level components to evaluate both the thermodynamic stability and spectral coherence of the system in one call.
