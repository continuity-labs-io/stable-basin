Write a PyTorch `nn.Module` named `SpatialCompressor` that serves as the bridge
between raw biological voxels and the temporal latent space.

Requirements:

1. The `forward` pass must accept the 5D tensor [Batch, Time, Channels, Depth,
   Height, Width].
2. Execute a 2D max-projection along the Depth axis (dim=3) to compress the
   volumetric data into a 4D tensor [Batch, Time, Channels, Height, Width].
3. Because standard Vision Transformers require 3 input channels, mathematically
   pad the 2-channel data by duplicating the first channel or appending a tensor
   of zeros, resulting in [Batch, Time, 3, Height, Width].
4. Reshape the batch and time dimensions temporarily to process all spatial
   frames in parallel.
5. Pass the frames through a frozen ViT-Base model from the `timm` library
   (e.g., `vit_base_patch16_224`). Ensure gradient calculation is disabled for
   the ViT to prevent memory exhaustion and preserve the pre-trained spatial
   geometry.
6. Return the compressed sequence tensor formatted as [Batch, Time, 768].
