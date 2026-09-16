Context: We are upgrading "Master Demo 1: The Bio-Blade Engine" to run on actual
human organoid pharmacological ground truth data. We need to create a new
dataloader for MaxWell Biosystems HD-MEA `.raw.h5` files, and then modify our
existing `src/demo/01_bio_blade_engine.py` to use it instead of the old 3Brain
`brw_dataloader`.

Task 1: Create `src/pipeline/ephys/maxwell_dataloader.py` Requirements:

1. Inherit from `torch.utils.data.Dataset` to create `MaxWellHDMEADataset`.
2. The `__init__` method accepts `file_path`, `sequence_length`, and
   `target_channels`.
3. Open the `.raw.h5` file natively using `h5py.File(self.file_path, 'r')`.
4. MaxWell files store the raw voltage trace array under keys like `'/sig'`,
   `'/routing/lsb'`, or `'/mapping/sig'`. Write dynamic logic (or a try-except
   block) to extract the correct 2D dataset. Its shape is typically
   `[Channels, TimeFrames]` (transposed compared to our old 3Brain format).
5. In `__getitem__`, slice the dataset to extract the requested temporal chunk
   (e.g., up to `sequence_length`) and the first `target_channels`.
6. Transpose the chunk so the final shape is `[Time, Channels]`.
7. **Crucial:** Apply robust Z-score normalization
   (`(chunk - chunk.mean()) / (chunk.std() + 1e-5)`) before returning it as a
   `torch.float32` tensor to ensure stable variance for the Mamba-2 engine.
8. Add a `__del__` method to ensure `self.file.close()` is called.

Task 2: Modify existing `src/demo/01_bio_blade_engine.py` Requirements:

1. **Imports:**
   - Remove `ContinuousHDMEADataset` import.
   - Import `MaxWellHDMEADataset` from `src.pipeline.ephys.maxwell_dataloader`.
2. **File Paths:**
   - Define paths for the new MaxWell data below the config variables in
     `main()`:
     `FILE_CONTROL = os.path.join(project_root, "data", "ephys", "Drug_2953_control.raw.h5")`
     `FILE_CRASH = os.path.join(project_root, "data", "ephys", "Drug_2953_50uM.raw.h5")`
3. **Data Ingestion (The Concatenation):**
   - Replace the `ContinuousHDMEADataset` instantiation block.
   - Attempt to instantiate
     `dataset_control = MaxWellHDMEADataset(FILE_CONTROL, sequence_length=SEQ_LEN, target_channels=TARGET_CHANNELS)`
     and similarly for `dataset_crash` using `FILE_CRASH`.
   - If successful, extract the first item `[0]` from each dataset, concatenate
     them along the time dimension `dim=0`, add a batch dimension
     `.unsqueeze(0)`, and move to device:
     `val_seq = torch.cat([control_chunk, crash_chunk], dim=0).unsqueeze(0).to(device)`.
   - **Fallback:** If files are missing, fallback to synthetic tensor:
     `val_seq = (torch.randn(1, SEQ_LEN * 2, TARGET_CHANNELS).abs() * 0.5).to(device)`,
     and simulate flatline `val_seq[:, EVENT_FRAME:, :] = 0.0`.
   - Extract
     `batch = val_seq[:, :SEQ_LEN, :].expand(BATCH_SIZE, -1, -1).contiguous()`
     to pass to the latency benchmark.
4. **Remove Old Flatline Simulation:**
   - Delete the code block `val_seq[:, EVENT_FRAME:, :] = 0.0` from the
     "Simulating Waddington Crash" section for the real data flow. The
     concatenation in the new data ingestion block inherently contains the true
     biological crash!
5. **Dashboard Updates:**
   - In `plot_bio_blade_dashboard()`, update the Panel 1 and Panel 2 vertical
     line labels to: `"50µM Diazepam Phase Transition"`.
   - Ensure interpolation logic maps KSM back to `SEQ_LEN * 2` since `val_seq`
     is now double the length.
