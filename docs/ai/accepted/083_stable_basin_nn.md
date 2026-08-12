I need to refactor the continuous-time physics masking logic from these files into a domain-agnostic, general-purpose PyTorch module for the `stable_basin.nn` package. Create a new file at `stable_basin/nn/mask_aware.py`.

Take the logic from `MaskAwareSSM` and `BiologicalCartridgeFusion` and combine them into a clean, reusable component called `MaskAwareMamba`.

Requirements:
1. Strip out all biological terminology (e.g., "cartridge", "epigenetics", "BiologicalCartridgeFusion").
2. The user should be able to initialize the layer simply with `d_model`, `d_state`, and `n_modalities`. Make the orthogonal routing dynamic based on these dimensions rather than hardcoding specific splits.
3. The `forward` pass should accept `x` (input sequence `[Batch, Time, Features]`) and `mask` (boolean/float mask indicating dropped sensors).
4. DO NOT ALTER the math for the "Physics Hack" (`dt_gated = dt_base * g_t + 1e-8`). This must remain mathematically identical.
5. Add clean, professional docstrings explaining how to use this for irregular time-series with NaNs (like IoT sensors or asynchronous financial ticks).

Files to tag in Composer: src/models/ssm/mask_aware_ssm.py, src/models/encoders/fusion.py, src/models/ssm/meld_engine.py
