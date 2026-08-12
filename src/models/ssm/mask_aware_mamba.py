import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
except ImportError:
    logging.warning("mamba_ssm is not installed. MaskAwareMamba will fall back to an Identity backbone.")
    Mamba2 = None

class MaskAwareMamba(nn.Module):
    """
    Mask-Aware Mamba-2 Engine for multimodal telemetry.
    
    Comparable to `MaskAwareSSM`, but uses a highly optimized Mamba-2 backbone 
    instead of a baseline ZOH continuous-time formulation. 
    
    Note on Heads: This module is specifically designed for Thermodynamic/Custom 
    loss functions, which is why it utilizes a dual-head architecture:
    - forward_head: standard causal forecasting (predicts t+1)
    - reverse_head: auto-encoding/reconstruction (reconstructs t)
    """
    def __init__(self, input_dim: int, d_model: int = 256, d_state: int = 64, mask_aware: bool = False):
        super().__init__()
        self.mask_aware = mask_aware
        
        if mask_aware:
            # double the input dim for concatenated sensor failure mask
            in_features = input_dim * 2
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            in_features = input_dim
            self.input_proj = nn.Linear(input_dim, d_model)

        if Mamba2 is not None:
            self.mamba = Mamba2(d_model=d_model, d_state=d_state)
        else:
            self.mamba = nn.Identity()

        # Standard causal forecasting head (predicts x_{t+1})
        self.forward_head = nn.Linear(d_model, input_dim)
        # Reverse head for thermodynamic loss (reconstructs x_t)
        self.reverse_head = nn.Linear(d_model, input_dim)

    def forward(self, x, mask=None, return_hidden=False):
        if self.mask_aware:
            if mask is None:
                mask = torch.isnan(x).float()
            x_safe = torch.nan_to_num(x, nan=0.0)
            x_in = torch.cat([x_safe, mask], dim=-1)
            h = self.input_proj(x_in)
        else:
            h = self.input_proj(x)

        hidden_states = self.mamba(h)

        # positive values in prediction
        pred_t_plus_1 = F.softplus(self.forward_head(hidden_states))
        reconstructed_t = F.softplus(self.reverse_head(hidden_states))

        if return_hidden:
            return pred_t_plus_1, reconstructed_t, hidden_states
        return pred_t_plus_1, reconstructed_t

    def get_hidden_states(self, x, mask=None):
        _, _, hidden_states = self.forward(x, mask=mask, return_hidden=True)
        return hidden_states


