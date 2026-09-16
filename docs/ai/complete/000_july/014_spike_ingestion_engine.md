Write a highly optimized Python module to ingest the
'mysteriousauthor/spikeprophecy-steinmetz' dataset. Relevant dataset
information:

- The dataset consists of 39 `.npy` files (`session_NNN.npy`) of shape
  `[n_units, n_bins]` containing `uint8` spike counts (50ms bins).
- A `metadata.json` provides per-session split boundaries (`train_end`,
  `val_end`) encoding a 70/15/15 train/val/test temporal split.
- Use `huggingface_hub.snapshot_download` to fetch the data instead of the
  `datasets` library directly.

Implement a PyTorch 'IterableDataset' that downloads and streams the 50ms binned
spike counts. The dataset must respect the temporal splits defined in
`metadata.json` to prevent leakage and yield sliding history windows of shape
(batch, time_steps, num_neurons). Prioritize execution speed and simplicity,
ensuring the host memory footprint remains flat regardless of the dataset size.

dependencies:

- python=3.10
- pytorch
- torchvision
- torchaudio
- pytorch-cuda=12.1
- huggingface_hub
- datasets
- numpy
- scipy
- pandas
- pip
- pip:
  - mamba-ssm
  - causal-conv1d
  - loguru

# requirements.txt

torch huggingface-hub datasets numpy scipy pandas mamba-ssm causal-conv1d loguru
