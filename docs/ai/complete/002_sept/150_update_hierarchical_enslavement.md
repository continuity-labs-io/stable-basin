Load src/demo/torx/hierarchical_enslavement.py.

Upgrade the evaluation sequence to explicitly measure hierarchical enslavement using Phase-Amplitude Coupling (PAC).

Import SpectralMetrics from src.metrics.

After simulating macro_B and micro_B, utilize SpectralMetrics.calculate_cfc_pac(macro_B, micro_B).

Replace the simple variance reduction calculation with this CFC metric, demonstrating that the slow macro-state prior is mathematically pacing the amplitude of the fast micro-states.

Update the console logging and the bottom dashboard panel's title to report the measured PAC value. Keep the language precise and practical.
