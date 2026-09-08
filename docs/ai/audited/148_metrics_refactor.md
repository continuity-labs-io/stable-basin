Context Preparation:
src/metrics/metrics.py
src/metrics/__init__.py
src/demo/01_hardware_scaling_proof.py
src/demo/05_flight_recorder_demo.py
src/harness/clinical_diagnostic_runner.py

Prompt:
We are refactoring our metrics engine to cleanly separate time-domain phase space calculations from frequency-domain spectral decoherence calculations. Please execute the following sequence:

1. Create a new file at `src/metrics/time_domain.py`. Extract the `calculate_dynamic_rank` function and the `ThermodynamicMetrics` class from `src/metrics/metrics.py` into this new file. Retain ONLY the time-domain methods in this class: `__init__`, `calculate_csd`, `calculate_ksm`, `calculate_hysteresis`, `calculate_lle`, `calculate_cka`, `calculate_epigenetic_dispersion`, and `extract_fedichev_macrostates`. Include all required imports (torch, torch.nn.functional, numpy, logging, pydmd, and settings).

2. Create a new file at `src/metrics/spectral.py`. Create a new class called `SpectralMetrics` in this file. Move the frequency-domain methods (`calculate_psd`, `calculate_plv`, `calculate_cfc_pac`) from the original `metrics.py` into this new class. Include all necessary imports (scipy.signal, numpy, torch, logging). Ensure the PyTorch tensor mathematics remain highly optimized and vector-driven.

3. Update `src/metrics/__init__.py` to act as the unified architectural facade. Import `ThermodynamicMetrics` from `.time_domain` and `SpectralMetrics` from `.spectral`, explicitly exposing both via the __all__ list.

4. Delete the original `src/metrics/metrics.py` file to permanently eliminate the monolith.

5. Update the import statements in `src/demo/01_hardware_scaling_proof.py`, `src/demo/05_flight_recorder_demo.py`, and `src/harness/clinical_diagnostic_runner.py`. Change instances of `from src.metrics.metrics import ThermodynamicMetrics` to `from src.metrics import ThermodynamicMetrics` to utilize the new, cleaner facade.

Keep all generated code calm, precise, and practical. Ensure the internal logic remains perfectly intact during the migration.
