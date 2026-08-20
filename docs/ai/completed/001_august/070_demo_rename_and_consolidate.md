Directive 1: Rename the Demos (Eradicate the Sci-Fi Fluff)We need the filenames in src/demo/ to map exactly to the empirical claims in your paper's Results section.Rename 01_bio_blade_engine.py $\rightarrow$ 01_hardware_scaling_proof.py(Claim: $\mathcal{O}(1)$ VRAM scaling and IOPS bypass on raw 20kHz HD-MEA telemetry).Rename 02_indestructible_edge.py $\rightarrow$ 02_fault_tolerant_imputation.py(Claim: MASR can dynamically impute massive sensor voids without hallucinating ghost noise).Rename 03_multimodal_diagnostic.py $\rightarrow$ 03_causal_interpretability.py(Claim: MambaLRPEpsilon traces macroscopic structural collapse back to an RNA root cause).Rename 04_masked_state_space_model.py $\rightarrow$ 04_gradient_stasis_proof.py(Claim: Mathematical unit test proving gradient preservation during a blackout).

Directive 2: Consolidate the SSM Spaghetti into meld_engine.py
Leave mask_aware_ssm.py and baseline_ssm.py entirely alone—your Waddington benchmark relies on them.

For the rest, delete neocortical_engine.py, spike_forecaster.py, and state_space_engine.py. Also, remove the inline DynamicMaskingEngine from your demos. Replace them all with a single, unified wrapper at src/models/ssm/meld_engine.py:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

class MeldEngine(nn.Module):
    """
    Unified Continuous-Time State Space Engine for MELD Demos.
    Handles standard forecasting, reverse time reconstruction, and mask-aware routing.
    """
    def __init__(self, input_dim: int, d_model: int = 256, d_state: int = 64, mask_aware: bool = False):
        super().__init__()
        self.mask_aware = mask_aware
        
        # If mask_aware is True, we double the input dim to concatenate the sensor failure mask
        in_features = input_dim * 2 if mask_aware else input_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        ) if mask_aware else nn.Linear(input_dim, d_model)

        # Core SSM Backbone
        if Mamba2 is not None:
            self.mamba = Mamba2(d_model=d_model, d_state=d_state)
        else:
            self.mamba = nn.Identity()

        # Forward head (predicts t+1) and Reverse head (reconstructs t for Thermodynamic loss)
        self.forward_head = nn.Linear(d_model, input_dim)
        self.reverse_head = nn.Linear(d_model, input_dim)

    def forward(self, x, return_hidden=False):
        if self.mask_aware:
            mask = torch.isnan(x).float()
            x_safe = torch.nan_to_num(x, nan=0.0)
            x_in = torch.cat([x_safe, mask], dim=-1)
            h = self.input_proj(x_in)
        else:
            h = self.input_proj(x)

        hidden_states = self.mamba(h)

        pred_t_plus_1 = F.softplus(self.forward_head(hidden_states))
        reconstructed_t = F.softplus(self.reverse_head(hidden_states))

        if return_hidden:
            return pred_t_plus_1, reconstructed_t, hidden_states
        return pred_t_plus_1, reconstructed_t

    def get_hidden_states(self, x):
        _, _, hidden_states = self.forward(x, return_hidden=True)
        return hidden_states

    def compute_attribution(self, x, target_time_step):
        """First-Order Taylor Decomposition for LRP/Interpretability."""
        x_req = x.clone().detach().requires_grad_(True)
        pred_t_plus_1, _ = self.forward(x_req)
        
        target_state_sum = pred_t_plus_1[:, target_time_step, :].sum()
        gradients = torch.autograd.grad(target_state_sum, x_req, retain_graph=True)[0]
        
        return x_req.detach() * gradients
```

Directive 3: Wire the Unified Engine to the Renamed Demos
Open your newly renamed demos and purge any remaining legacy imports or inline model classes. Standardize them so they all call your unified engine:

In 01_hardware_scaling_proof.py & 03_causal_interpretability.py:

```
from src.models.ssm.meld_engine import MeldEngine

# Standard processing
model = MeldEngine(input_dim=TARGET_CHANNELS, d_model=256, mask_aware=False).to(device)
```

In 02_fault_tolerant_imputation.py:

```
from src.models.ssm.meld_engine import MeldEngine

# Catastrophic hardware failure processing
model = MeldEngine(input_dim=114, d_model=256, mask_aware=True).to(device)
```
