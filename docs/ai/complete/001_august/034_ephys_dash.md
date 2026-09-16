Context: We are creating a master execution script to prototype the
continuous-time Mamba-2 engine on raw 1,024-channel HD-MEA data. We have
`brw_dataloader.py` which ingests raw Zenodo 3Brain data, the `SpikeForecaster`
(Mamba-2), `ThermodynamicMetrics` (PyDMD), and `MambaLRPEpsilon`.

Task: Create a new script `src/demo/8_ephys_demo.py`.

1. **Setup & Ingestion:** Initialize the `ContinuousHDMEADataset` using a dummy
   `.brw` file. Use a batch size of 1 and a sequence length of 10,000 (500ms at
   20kHz).
2. **Burn-in Training:** Pass the chunked tensor `[1, 10000, 1024]` into the
   `SpikeForecaster`. Optimize it using `MeldLoss` for 10 iterations to burn in
   the biological standing wave dynamics. Track the peak VRAM allocation using
   `HardwareMonitor`.
3. **The Waddington Crash:** On a validation sequence, mathematically inject a
   "variance explosion" (simulate a seizure or membrane rupture) starting
   exactly at the 5,000th frame.
4. **Thermodynamic Extraction:** Pass the crashed sequence through the model,
   extract the hidden states, and run `ThermodynamicMetrics.calculate_ksm` to
   generate the Koopman Stability Metric array.
5. **Attribution:** Run `MambaLRPEpsilon` targeting the exact 5,000th frame to
   extract the `[1, 10000, 1024]` relevance tensor showing which electrodes
   drove the crash.
6. **The Dashboard:** Use Matplotlib to generate a publication-ready, 4-panel
   dark-mode figure named `8_ephys_demo.png`.
   - Panel 1: 2D Heatmap of the raw 20kHz voltage traces (subsample 64 channels
     for visual clarity).
   - Panel 2: The flat O(1) VRAM Hardware Monitor chart.
   - Panel 3: The PyDMD KSM metric plotting the sudden collapse of biological
     stability at frame 5,000.
   - Panel 4: 2D Heatmap of the MambaLRP attribution relevance, proving the AI
     isolated the localized root cause of the crash.
