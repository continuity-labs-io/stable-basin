Context Files to Load:

src/models/ssm/triton_fused_scan.py

src/models/ssm/dynamic_integrator.py

(Create Target) src/models/ssm/async_ssm.py

Raw Prompt to Execute:

We need a hardware-agnostic wrapper that executes our asynchronous SSM on Mac
(using PyTorch) and automatically switches to the Triton kernel on NVIDIA GPUs.

Create a new module `src/models/ssm/async_ssm.py`.

1. Write an `AsyncMaskAwareSSM` PyTorch Module. In its
   `__init__(self, dim, d_state)`, initialize the core continuous-time
   parameters: `self.A_log` (nn.Parameter) and `self.B_proj` (nn.Parameter).
2. In its `forward(self, events, event_mask)` pass, check if `HAS_TRITON` is
   True (imported from `src.models.ssm.triton_fused_scan`) AND
   `events.device.type == "cuda"`.
3. If True, execute the high-performance
   `MaskAwareFusedScan.apply(events, event_mask, self.A_log, self.B_proj)`.
4. If False (e.g., on macOS `mps` or `cpu`), fall back to executing pure PyTorch
   logic. To keep things DRY, you can move the pure PyTorch computation from
   `DynamicIntegrator` into a helper method here (using `self.A_log` and
   `self.B_proj`), or refactor `DynamicIntegrator` to accept the weights
   dynamically instead of defining its own parameters.
5. Return the output hidden state of shape `[Batch, Max_Events, Dim, D_State]`.
