For the clinical_diagnostic test we currently have a custom KSM threshold 
of 0.85 (hardcoded) vs 0.95 for ssms  because mamba "jumps through time" and that triggers KSM drops. The benefit of the following work would be to remove the hardcoded threshold.
 
We can constrain the Mamba block so it acts more like a rigid Linear Time-Invariant (LTI) system during homeostasis, while retaining its capacity to adapt during a crash. In PyTorch, `nn.Linear` layers have both a `weight` and a `bias`. If we initialize the *weights* of the $B$, $C$, and $\Delta t$ projections to be vanishingly small, their outputs will be dominated almost entirely by their static *biases*. This essentially initializes Mamba as a stable LTI system. It will only engage its chaotic LTV data-dependency when the massive gradients of the biological crash force the weights to adapt.

Please implement in `src/models/ssm/masr_mamba.py`:

Locate the `PyTorchMambaMASR.__init__` method.

Immediately after the data-dependent parameter projections (`self.B_proj`, `self.C_proj`, `self.dt_proj`) are defined (and before the `dt_proj.bias` initialization), add the following structural regularizations:

```python
# --- Architectural Regularization (LTI Prior) ---
# Dampen initial data-dependent chaos to stabilize the baseline Koopman Stability Metric.
# By shrinking the weights, the outputs are dominated by the static biases, 
# allowing Mamba to behave like a stable LTI system until forced to adapt.
nn.init.normal_(self.B_proj.weight, std=0.01)
nn.init.normal_(self.C_proj.weight, std=0.01)
nn.init.normal_(self.dt_proj.weight, std=0.01)
```
