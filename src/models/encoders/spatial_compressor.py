from jaxtyping import jaxtyped, Float
from beartype import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import timm

import logging

logger = logging.getLogger(__name__)


class SpatialCompressor(nn.Module):
    """
    SpatialCompressor serves as the bridge between raw biological voxels and the temporal latent
        space.
    """

    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()

        # 5. Load pre-trained ViT without the classification head (num_classes=0 outputs pooled
        # features)
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Ensure gradient calculation is disabled for the ViT to prevent memory exhaustion
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[torch.Tensor, "B T C D H W"]) -> Float[torch.Tensor, "B T 768"]:
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

        # 3. Mathematically pad the data to 3 channels for the ViT backbone
        if C == 1:
            zeros = torch.zeros((B, T, 2, H, W), dtype=x_proj.dtype, device=x_proj.device)
            x_padded = torch.cat([x_proj, zeros], dim=2)
        elif C == 2:
            zeros = torch.zeros((B, T, 1, H, W), dtype=x_proj.dtype, device=x_proj.device)
            x_padded = torch.cat([x_proj, zeros], dim=2)
        elif C == 3:
            x_padded = x_proj
        else:
            raise ValueError(f"SpatialCompressor expects 1, 2, or 3 channels. Got {C} channels.")

        # 4. Process frames sequentially over the time dimension to maintain O(N) VRAM
        features = []
        for t in range(T):
            x_t = x_padded[:, t]  # Shape: [Batch, 3, Height, Width]

            # Interpolate to 224x224 since the vit_base_patch16_224 requires 224x224 geometry
            if H != 224 or W != 224:
                x_t = F.interpolate(x_t, size=(224, 224), mode="bilinear", align_corners=False)

            # 5. Pass the batch of frames through the frozen ViT-Base model
            if x_t.requires_grad:
                feat_t = cp.checkpoint(self.vit, x_t, use_reentrant=False)
            else:
                feat_t = self.vit(x_t)  # Shape: [Batch, 768]
                
            features.append(feat_t)

        # 6. Return the compressed sequence tensor formatted as [Batch, Time, 768]
        out = torch.stack(features, dim=1)

        return out
