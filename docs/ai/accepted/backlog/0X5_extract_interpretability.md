Files to tag in Composer: src/metrics/mamba_lrp.py, src/metrics/diagnostic_engine.py

I need to extract the exact Layer-wise Relevance Propagation (LRP) logic into a standalone interpretability tool for the `stable_basin.interpretability` module. Create a file at `stable_basin/interpretability/explainer.py`.

Please refactor `MambaLRPEpsilon` and `ThermodynamicDiagnosticEngine` into a clean, general-purpose set of classes.

Requirements:
1. The tool must be decoupled from `SpikeForecaster`. It should accept ANY generic PyTorch model that exposes a `get_hidden_states` method.
2. Modify the initialization to accept generic arguments for the input and output linear layers (so the user can pass `input_proj_layer_name` and `output_proj_layer_name` dynamically) and use `getattr` to fetch them during the forward pass.
3. Remove all hardcoded biological feature names (e.g., "Sigma_PC", "Psi_TP53") from the diagnostic engine. Allow the user to optionally pass a list of `feature_names` during initialization for the causal trace report.
4. DO NOT alter the underlying LRP-epsilon math, the epsilon stabilizer, or the O(N) backward unrolling loop. That is the secret sauce.
