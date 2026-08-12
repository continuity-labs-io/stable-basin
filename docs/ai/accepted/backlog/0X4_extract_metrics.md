I am extracting the latent thermodynamic metrics from my biological codebase into a generic early-warning toolkit for ML engineers. I want to put this in `stable_basin/metrics/radar.py`.

Please refactor `ThermodynamicMetrics` into a new, clean class called `EarlyWarningRadar`.

Requirements:
1. Remove all biological and aging terminology from the codebase (e.g., Fedichev, epigenetic, Waddington, organoid). Replace log messages with generic systems terminology (e.g., "Latent manifold collapse detected").
2. Keep the core mathematical methods: `calculate_ksm` (PyDMD Koopman eigenvalues), `calculate_csd` (Critical Slowing Down), and `calculate_lle` (Lyapunov exponents).
3. Move the highly biological functions (`calculate_epigenetic_dispersion`, `extract_fedichev_macrostates`) OUT of this file. Do not delete them; create a new file in `examples/aging_research/fedichev.py` and put them there.
4. Remove dependencies on a global `settings.py`. Pass configuration variables like `window_size`, `alpha`, and `beta` directly into the `__init__`.
5. Add clear docstrings explaining that a KSM score dropping below 0.9 indicates imminent structural collapse of a model's latent space, making this useful for DevOps or quant trading model monitoring.

Files to tag in Composer: src/metrics/metrics.py
