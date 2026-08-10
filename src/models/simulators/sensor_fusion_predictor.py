import torch
import torch.nn as nn
from src.models.encoders.fusion import BiologicalCartridgeFusion
from src.models.ssm.baseline_ssm import BaselineSSM
from src.models.ssm.mask_aware_ssm import MaskAwareSSM
from src.models.attention.baseline_transformer import BaselineTransformer
from src.models.rnn.gru_d import GRUDModel
from src.models.rnn.ode_rnn import ODERNNModel

from enum import Enum

class SSMType(str, Enum):
    """The type of state-space model or baseline to instantiate."""
    BASELINE = "baseline"
    """Zero-padded SSM baseline."""
    FORWARD_FILL = "forward_fill"
    """Forward-fill (hold-last-value) SSM baseline."""
    MASK_CONCAT = "mask_concat"
    """Mask-concatenated SSM baseline."""
    TRANSFORMER = "transformer"
    """Causal Transformer baseline."""
    MASK_AWARE = "mask_aware"
    """MASR (Mask-Aware State-Space Representation) model."""
    GRU_D = "gru_d"
    """GRU-D model."""
    ODE_RNN = "ode_rnn"
    """ODE-RNN model using torchdiffeq."""

class SensorFusionPredictor(nn.Module):
    def __init__(
        self, ssm_type: str, d_cartridge: int = 30, n_modalities: int = 2, d_model: int = 64
    ):
        super().__init__()
        self.ssm_type = ssm_type
        
        # For mask_concat, we concatenate the 2D mask to the 30D raw data
        if ssm_type == "mask_concat":
            self.fusion = BiologicalCartridgeFusion(d_cartridge + n_modalities, n_modalities, d_model)
        else:
            self.fusion = BiologicalCartridgeFusion(d_cartridge, n_modalities, d_model)

        if ssm_type in ["baseline", "forward_fill", "mask_concat"]:
            self.ssm = BaselineSSM(d_model)
        elif ssm_type == "mask_aware":
            self.ssm = MaskAwareSSM(d_model)
        elif ssm_type == "transformer":
            self.ssm = BaselineTransformer(d_model)
        elif ssm_type == "gru_d":
            self.ssm = GRUDModel(d_model)
        elif ssm_type == "ode_rnn":
            self.ssm = ODERNNModel(d_model)
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.readout = nn.Linear(d_model, 1)

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        if self.ssm_type == "forward_fill":
            # Apply Forward-Fill (Hold-Last-Value) to the sparse modality (indices 20-29)
            x_ff = x_raw.clone()
            batch_size, seq_len, _ = x_raw.shape
            last_known = torch.zeros_like(x_raw[:, 0, 20:])
            for t in range(seq_len):
                m_t = mask[:, t, 1:2]
                current_x = x_raw[:, t, 20:]
                last_known = torch.where(m_t == 1.0, current_x, last_known)
                x_ff[:, t, 20:] = last_known
            
            latent_x, latent_gate = self.fusion(x_ff, mask)
            h = self.ssm(latent_x)
            
        elif self.ssm_type == "mask_concat":
            # Concatenate mask directly to the features
            x_concat = torch.cat([x_raw, mask], dim=-1)
            latent_x, latent_gate = self.fusion(x_concat, mask)
            h = self.ssm(latent_x)
            
        elif self.ssm_type in ["gru_d", "ode_rnn"]:
            # Compute time deltas dynamically based on the mask
            # For every step where mask == 0, delta_t increments.
            # Where mask == 1, delta_t resets to 0.
            B, L, _ = x_raw.shape
            delta_t = torch.zeros(B, L, 1, device=x_raw.device)
            current_delta = torch.zeros(B, 1, device=x_raw.device)
            
            for t in range(L):
                m_t = mask[:, t, 1:2]
                current_delta = current_delta + 1.0
                delta_t[:, t, :] = current_delta
                current_delta = torch.where(m_t == 1.0, torch.zeros_like(current_delta), current_delta)
                
            latent_x, latent_gate = self.fusion(x_raw, mask)
            h = self.ssm(latent_x, delta_t)

        else:
            latent_x, latent_gate = self.fusion(x_raw, mask)
            if self.ssm_type == "baseline":
                h = self.ssm(latent_x)
            elif self.ssm_type == "mask_aware":
                h = self.ssm(latent_x, latent_gate)
            elif self.ssm_type == "transformer":
                h = self.ssm(latent_x)

        return self.readout(h)
