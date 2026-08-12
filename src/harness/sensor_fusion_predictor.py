import torch
import torch.nn as nn
from src.models.encoders.orthogonal_modality_encoder import OrthogonalModalityEncoder
from src.models.ssm.baseline_ssm import BaselineSSM
from src.models.ssm.mask_aware_ssm import MaskAwareSSM
from src.models.ssm.mask_aware_mamba import MaskAwareMamba
from src.models.attention.baseline_transformer import BaselineTransformer
from src.models.rnn.gru_d import GRUDModel
from src.models.rnn.ode_rnn import ODERNNModel

from typing import Optional
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
    MASK_AWARE_MAMBA = "mask_aware_mamba"
    """Mask-Aware Mamba-2 model."""
    GRU_D = "gru_d"
    """GRU-D model."""
    ODE_RNN = "ode_rnn"
    """ODE-RNN model using torchdiffeq."""

class SensorFusionPredictor(nn.Module):
    def __init__(
        self, ssm_type: str, modality_dims: list[int] = None, d_model: int = 64, out_dim: int = 1
    ):
        """
        Initializes the SensorFusionPredictor.

        Args:
            ssm_type (str): The type of State-Space Model or baseline to instantiate 
                (e.g., 'mask_aware', 'mask_aware_mamba', 'transformer', 'baseline').
            modality_dims (list[int]): A list of dimensions for each modality.
                Defaults to [20, 10] (20-dim dense, 10-dim sparse) if not provided.
            d_model (int): The hidden dimension (latent space) for the state space model.
        """
        super().__init__()
        self.ssm_type = ssm_type

        # Default to the Waddington dataset dimensions if not provided
        if modality_dims is None:
            modality_dims = [20, 10]
            
        d_sensor_total = sum(modality_dims)
        num_modalities = len(modality_dims)

        if ssm_type == "mask_concat":
            # The input dimension is inflated by the mask size
            self.fusion = OrthogonalModalityEncoder(
                d_in=d_sensor_total + num_modalities, 
                modality_dims=modality_dims, 
                d_model=d_model
            )
        else:
            self.fusion = OrthogonalModalityEncoder(
                d_in=d_sensor_total, 
                modality_dims=modality_dims, 
                d_model=d_model
            )

        if ssm_type in ["baseline", "forward_fill", "mask_concat"]:
            self.ssm = BaselineSSM(d_model)
        elif ssm_type == "mask_aware":
            self.ssm = MaskAwareSSM(d_model)
        elif ssm_type == "mask_aware_mamba":
            self.ssm = MaskAwareMamba(input_dim=d_model, d_model=d_model, mask_aware=True)
        elif ssm_type == "transformer":
            self.ssm = BaselineTransformer(d_model)
        elif ssm_type == "gru_d":
            self.ssm = GRUDModel(d_model)
        elif ssm_type == "ode_rnn":
            self.ssm = ODERNNModel(d_model)
        else:
            raise ValueError(f"Unknown ssm_type: {ssm_type}")

        self.readout = nn.Linear(d_model, out_dim)

    def forward(self, x_raw: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.ssm_type == "forward_fill":
            # Apply Forward-Fill (Hold-Last-Value) to the sparse modality
            # (indices 20-29)
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
            # Compute time deltas dynamically based on the mask For every step
            # where mask == 0, delta_t increments. Where mask == 1, delta_t
            # resets to 0.
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
            elif self.ssm_type == "mask_aware_mamba":
                h = self.ssm.get_hidden_states(latent_x, mask=latent_gate)
            elif self.ssm_type == "transformer":
                h = self.ssm(latent_x)

        preds = self.readout(h)
        return preds, h

    def get_hidden_states(self, x, mask=None):
        """Convenience method for cleanly extracting the thermodynamic manifold."""
        _, hidden_states = self.forward(x, mask=mask)
        return hidden_states
