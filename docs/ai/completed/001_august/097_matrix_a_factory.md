# Matrix A Factory

Refactor the A matrix initialization across the State Space Models to use a Factory pattern.

1. Create a factory (`src/models/ssm/physics/matrix_a_factory.py`) that generates the A matrix initializer.
2. The factory will strictly support `"random"` and `"log_spaced"` strategies.
3. Rename the previous `BiologicalSpectrogramInit` to `LogSpacedAInit` and move it to the factory.
4. Hook this factory into the existing SSM variants (`baseline_ssm.py`, `masr_ssm.py`, `async_ssm.py`, `dynamic_integrator.py`, `masr_mamba.py`) to replace their hardcoded random A matrix initializations.
5. The SSM variants will use their existing shapes (`[d_model]` vs `[d_model, d_state]`) and pass that shape tuple to the factory to correctly broadcast.
6. The Harness level (`SensorFusionPredictor`, etc.) will accept an `a_init_type` string argument (defaulting to `"random"`) and pass it down into the SSM models.
