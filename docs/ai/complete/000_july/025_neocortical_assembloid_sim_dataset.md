Create a new PyTorch IterableDataset class named `NeocorticalAssembloidDataset`
in `src/pipeline/neocortical_assembloid_dataloader.py`. Context: We are
simulating a multi-region thalamocortical loop assembloid generating multi-modal
time-series data (HD-MEA voltage traces + spatial optical markers + sparse RNA
snapshots).

Requirements:

1. Initialize with `time_steps` (default 1000 frames), `num_channels` (default
   256 electrode channels), and `latent_dim` (default 114 to match the MELD
   Trifecta: 100D Sigma + 12D Psi + 2D Omega).
2. Implement `__iter__` to yield continuous batches modeling a 3D living
   assembloid under homeostasis, transitioning into a slow metabolic crash at
   70% of the sequence length.
3. Inject a progressive variance explosion and desynchronization of macroscopic
   standing waves starting at the crash threshold.
4. Return a dictionary or tuple containing the multi-modal sequence tensor
   [Batch, Time, 114] and an explicit binary health label [Batch, Time] (1 for
   homeostasis, 0 for collapse).

upsert to environment.yml:

dependencies:

- python=3.11
- pip
- pandas
- pyarrow
- pytorch
- numpy
- tifffile
- scikit-learn
- matplotlib
- pip:
  - torch
  - mamba-ssm-macos
  - causal-conv1d
  - timm
  - pydantic-settings

requirements: fsspec matplotlib numpy pandas pyarrow requests scikit-learn timm
torch mamba-ssm-macos causal-conv1d pydantic-settings
