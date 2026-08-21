import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.config import settings


class TopoEncoder(nn.Module):
    """
    Ingests continuous E-field flow and uses Mamba-2 to extract the
    macroscopic geometric shape (Dynamic Attractor Basin) into a fixed latent vector.
    
    TODO: Hook this up to process continuous electrophysiological standing waves (e.g., from 
    SpikeProphecyDataset or OmegaBioelectricLoader) to provide spatial priors to the main 
    Thermodynamic State Space Model. Currently isolated.
    """

    def __init__(
        self, d_model=settings.MAMBA_D_MODEL, d_state=settings.MAMBA_D_STATE, d_conv=4, expand=2
    ):
        super().__init__()

        # 2D Convolutional frontend to compress 64x64 spatial dimensions to d_model
        # Input shape per frame: [2, 64, 64]
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=8, stride=4, padding=2),  # -> [64, 16, 16]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=4, stride=2, padding=1),  # -> [192, 8, 8]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, d_model, kernel_size=8, stride=1, padding=0),  # -> [d_model, 1, 1]
            nn.Flatten(),  # -> [d_model]
        )

        # Mamba block to process sequence over time
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # MLP projection head with LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x, return_hidden=False):
        """
        Args:
            x: Tensor of shape [Batch, Time, 2, 64, 64] representing the E-field flow
            return_hidden: Boolean, if True, returns the full sequence of hidden states

        Returns:
            projected_latent: Tensor of shape [Batch, 768]
            (optional) hidden_states: Tensor of shape [Batch, Time, 768]
        """
        B, T, C, H, W = x.shape

        # Fold batch and time dimensions to process frames through Conv2d
        x = x.view(B * T, C, H, W)

        # Encode spatial features
        spatial_features = self.spatial_encoder(x)  # Shape: [B*T, d_model]

        # Unfold back to sequence format
        sequence = spatial_features.view(B, T, -1)  # Shape: [B, Time, d_model]

        # Process through Mamba to capture continuous thermodynamic loops
        hidden_states = self.mamba(sequence)  # Shape: [B, Time, d_model]

        # Extract the hidden state of the final time step
        final_state = hidden_states[:, -1, :]  # Shape: [B, d_model]

        # Project to final latent space
        projected_latent = self.mlp_head(self.norm(final_state))  # Shape: [B, d_model]

        if return_hidden:
            return projected_latent, hidden_states
        return projected_latent
