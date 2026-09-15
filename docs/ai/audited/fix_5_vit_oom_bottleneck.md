# Fix 5: Fix ViT OOM Bottleneck in SpatialCompressor

**Severity:** High

## Description
Passing B*T images simultaneously through a Vision Transformer causes massive VRAM spikes and inevitable Out-Of-Memory errors.

## AI Execution Plan
Develop a plan to refactor SpatialCompressor.forward to implement a mini-batching loop over the Time dimension or use gradient checkpointing. Get it reviewed, implement, ensure tests pass, and commit.
