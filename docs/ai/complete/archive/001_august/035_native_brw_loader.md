Context: The `ContinuousHDMEADataset` in `src/pipeline/ephys/brw_dataloader.py`
currently uses `spikeinterface`, which is crashing with a divide-by-zero error
on our 12GB Zenodo `.brw` file due to missing metadata headers. We need to
bypass it and read the HDF5 structure natively.

Task: Rewrite `ContinuousHDMEADataset` to use `h5py` directly.

1. Remove `spikeinterface` imports. Import `h5py`.
2. In `__init__`, open the `.brw` file in read-only mode using `h5py.File`. The
   raw 20kHz continuous traces in a 3Brain .brw file are stored as a massive 1D
   or 2D array under the key path: `/3BData/Raw`.
3. Extract the `Raw` dataset shape. Calculate the total frames assuming 4096
   total channels.
4. In `__getitem__`, slice the `Raw` HDF5 dataset to extract the requested chunk
   of frames.
5. Spatial Subsampling: Take only the first `target_channels` (default 1024).
6. **Crucial Scaling Step:** The raw data is stored as uncalibrated integers.
   Apply a robust Z-score normalization to the chunk
   (`(chunk - chunk.mean()) / (chunk.std() + 1e-5)`) before returning it as a
   `torch.float32` tensor. This ensures the Mamba-2 engine receives stable
   `[-3, +3]` variance regardless of the file's raw microvolt scale.
