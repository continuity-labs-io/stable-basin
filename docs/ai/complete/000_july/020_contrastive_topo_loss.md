Update the existing `src/models/meld_loss.py` file to include a new custom loss
function `TopoContrastiveLoss(nn.Module)`. Context: We are proving Ephaptic
Lock-in by aligning the topological shape of the brain's continuous standing
wave with the visual stimuli it is experiencing, similar to CLIP's image-text
alignment.

Requirements:

1. Initialize with a learnable temperature parameter `logit_scale` (initialized
   to `np.log(1 / 0.07)`).
2. The `forward` pass accepts two tensors: `lfp_latents` (shape `[Batch, 768]`)
   and `vision_latents` (shape `[Batch, 768]`).
3. L2-normalize both sets of latent vectors along the feature dimension.
4. Calculate the cosine similarity matrix between all LFP topologies and all
   visual stimuli in the batch:
   `logits = (lfp_latents @ vision_latents.T) * self.logit_scale.exp()`.
5. Calculate the symmetric Cross-Entropy loss (InfoNCE) for both the rows and
   the columns (ensuring the LFP wave predicts the image, and the image predicts
   the LFP wave) and return the average.
6. Return the scalar loss and a detached dictionary for telemetry (logging the
   loss and temperature value).
