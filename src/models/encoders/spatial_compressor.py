import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import logging

logger = logging.getLogger(__name__)


class SpatialCompressor(nn.Module):
    """
    SpatialCompressor serves as the bridge between raw biological voxels and the temporal latent space.
    """

    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()

        # 5. Load pre-trained ViT without the classification head (num_classes=0 outputs pooled features)
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Ensure gradient calculation is disabled for the ViT to prevent memory exhaustion
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 6D tensor of shape [Batch, Time, Channels, Depth, Height, Width]
                              where Channels=2 and spatial dimensions are typically 128x128.

        Returns:
            torch.Tensor: Compressed sequence tensor of shape [Batch, Time, 768]
        """
        # Ensure we are in eval mode for the frozen backbone
        self.vit.eval()

        # 1. Accept the 5D/6D tensor
        B, T, C, D, H, W = x.shape

        # 2. 2D max-projection along the Depth axis (dim=3)
        # Results in shape: [Batch, Time, Channels, Height, Width]
        x_proj, _ = torch.max(x, dim=3)

        # 3. Mathematically pad the 2-channel data by appending a tensor of zeros
        # Resulting in [Batch, Time, 3, Height, Width]
        zeros = torch.zeros((B, T, 1, H, W), dtype=x_proj.dtype, device=x_proj.device)
        x_padded = torch.cat([x_proj, zeros], dim=2)

        # 4. Reshape batch and time dimensions temporarily to process all spatial frames in parallel
        x_flat = x_padded.view(B * T, 3, H, W)

        # Interpolate to 224x224 since the vit_base_patch16_224 requires 224x224 geometry
        if H != 224 or W != 224:
            x_flat = F.interpolate(x_flat, size=(224, 224), mode="bilinear", align_corners=False)

        # 5. Pass the frames through the frozen ViT-Base model
        with torch.no_grad():
            features = self.vit(x_flat)  # Shape: [B*T, 768]

        # 6. Return the compressed sequence tensor formatted as [Batch, Time, 768]
        out = features.view(B, T, -1)

        return out
