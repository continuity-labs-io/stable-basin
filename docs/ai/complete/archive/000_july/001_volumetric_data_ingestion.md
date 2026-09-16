Write a PyTorch Dataset class named `AOLLSMDataset` designed to ingest a
directory of sequential TIFF stacks.

Requirements:

1. The class must iterate through 199 temporal directories (e.g., `stack0000` to
   `stack0198`).
2. Within each temporal step, load the volumetric TIFF files corresponding to
   `ch1` and `ch2`.
3. Stack these two physical channels together to form a multi-channel volume.
4. Apply a central spatial crop to extract a highly dense 128x128x128 voxel
   region from the center of the volume.
5. The `__getitem__` method should return a single, continuous 5D sequence
   tensor representing the entire time-series for a given sample, formatted
   exactly as [Time, Channels, Depth, Height, Width], which evaluates to [199,
   2, 128, 128, 128].
6. Utilize the `tifffile` library for optimal memory management during I/O
   operations.
