Create a new file named `brw_dataloader.py` for the Stable Basin Benchmark
repository.

I need a PyTorch `Dataset` and `DataLoader` pipeline designed to ingest
continuous high-density electrophysiology data from raw 3Brain `.brw` files. The
goal is to simulate the data topology of a 1,024-channel MaxOne CMOS array
streaming at 20kHz, outputting chunked sequence tensors ready for a
continuous-time state-space model (like Mamba-2).

Use `spikeinterface.extractors` to handle the underlying HDF5/brw file reading.

Requirements for the `ContinuousHDMEADataset` class:

1. Inherit from `torch.utils.data.Dataset`.
2. The `__init__` method should accept `brw_file_path`, `sequence_length`
   (default 10000), and `target_channels` (default 1024). Use
   `spikeinterface.extractors.read_3brain` to load the recording lazily. Extract
   the total number of frames and the sampling rate.
3. The `__len__` method should return the total number of non-overlapping chunks
   (total frames divided by sequence length).
4. The `__getitem__` method should fetch the specific temporal chunk of traces
   using the recording's `get_traces` method. Subsample the spatial dimension by
   keeping only the first `target_channels`. Return a PyTorch float32 tensor of
   shape `[Sequence_Length, Channels]`.
5. Add an `if __name__ == "__main__":` block that instantiates the dataset using
   a dummy file named `example.brw` and a `DataLoader` with batch size 4.
   Iterate over the first batch and print out the resulting tensor shape to
   confirm it yields `[Batch, Sequence_Length, Channels]`.

Write clean, production-ready Python code with clear docstrings and comments
explaining the temporal chunking and spatial subsampling steps. Include a
default directory to read the data as ]repo-root]/data/ephys.

# requirements.txt

torch numpy h5py spikeinterface
