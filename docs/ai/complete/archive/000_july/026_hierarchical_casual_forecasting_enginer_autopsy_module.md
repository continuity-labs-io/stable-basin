Create a PyTorch module `NeocorticalEngine` in
`src/models/neocortical_engine.py`. Context: This module unifies the Mamba-2
sequential forecaster with a thermodynamic state-space bridge and an
interpretability attribution hook.

Requirements:

1. Inherit from `nn.Module`. Import `Mamba2` from `mamba_ssm`.
2. Architecture:
   - Project input 114-D multimodal state vectors to `d_model=768` via a linear
     layer.
   - Pass through a stack of two `Mamba2` blocks with state dimension
     `d_state=64` to capture both local spike dynamics and slow macro-organoid
     phase transitions.
   - Include a forward predictive coding head predicting the state at t+1 and a
     reverse reconstruction head for time-reversal validation.
3. Integrate a method `compute_attribution(self, x, target_time_step)` that
   executes First-Order Taylor Decomposition (Input \* Gradient) on the
   predicted state vector at `target_time_step`, returning an attribution matrix
   of shape `[Time, Features]` to pinpoint failure root causes.
