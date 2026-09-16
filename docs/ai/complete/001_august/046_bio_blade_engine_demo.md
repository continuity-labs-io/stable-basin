Context: We are consolidating the Project MELD repository from a prototyping
sandbox into a "Director's Cut" of three master demonstrations. This task builds
"Master Demo 1: The Bio-Blade Engine", which serves as our core hardware
infrastructure pitch to VCs and systems engineers, proving that continuous 20kHz
biological telemetry must be processed via State Space Models natively on the
edge.

Task: Create a new script `src/demo/01_bio_blade_engine.py` by merging the core
concepts from `src/demo/raw/3_telemetry_matrix_bench.py` and
`src/demo/raw/8_ephys_demo.py`.

Requirements:

1. **Imports & Setup:**

   - Import `torch`, `time`, `os`, `numpy`, and `matplotlib.pyplot` (using
     `matplotlib.use("Agg")`).
   - Import `ContinuousHDMEADataset` from `src.pipeline.ephys.brw_dataloader`.
   - Import `SpikeForecaster` from `src.models.spike_forecaster`.
   - Import `ThermodynamicMetrics` from `src.metrics.metrics`.
   - Import `HardwareMonitor` from `src.metrics.hardware_monitor`.
   - Import `get_optimal_device` from `src.utils.device`.
   - Define the `format_bytes` and `format_bandwidth` utility functions (from
     `3_telemetry_matrix_bench.py`) at the top of the file to print clean,
     human-readable network statistics.
   - Configure a standard `logging` setup to output INFO to the console.

2. **The Execution Parameters (DEMO KNOBS):**

   - At the top of `main()`, define: `BATCH_SIZE = 8` `SEQUENCE_LENGTH_MS = 500`
     `CRASH_INJECTION_MS = 250` `SAMPLING_RATE_HZ = 20000`
     `TARGET_CHANNELS = 1024`
   - Calculate `SEQ_LEN` and `EVENT_FRAME` mathematically from these.

3. **Data Ingestion (The Reality Check):**

   - Initialize device using
     `get_optimal_device(allow_mps=False, verbose=True)`.
   - Instantiate the `ContinuousHDMEADataset` targeting
     `data/ephys/example.brw`. If the file is missing or throws an exception,
     gracefully fallback to a synthetic tensor of
     `torch.randn(BATCH_SIZE, SEQ_LEN, TARGET_CHANNELS).abs() * 0.5`. Extract a
     single batch.

4. **The Latency & Bandwidth Benchmark (The CLI Output):**

   - Initialize the `SpikeForecaster` (d_model=256, d_state=64) and set to
     `.eval()`.
   - Run a fast benchmarking loop simulating the inference of 10 sequential
     batches.
   - Measure the exact `time.perf_counter()` latency of the forward pass
     `model(batch)`.
   - Calculate the theoretical Cloud Bandwidth equivalent if this data were
     streamed to a Kubernetes cluster:
     `(BATCH_SIZE * SEQ_LEN * TARGET_CHANNELS * 4 bytes * 8 bits) / latency_sec / 1e9`
     for Gbps.
   - Print a stark ASCII benchmark table to the console explicitly comparing the
     **Local Edge Inference Latency (ms)** against the **Theoretical Cloud
     Bandwidth Required (Gbps)**.
   - Add a concluding print statement: _"CONCLUSION: Edge-compute prevents AWS
     ingress throttling."_

5. **The Waddington Crash & PyDMD Extraction:**

   - Clone the final batch to `val_seq`. Simulate a true biological flatline
     (necrosis) by setting `val_seq[:, EVENT_FRAME:, :] = 0.0`.
   - Pass `val_seq` through the model to extract `hidden_states`.
   - Initialize `ThermodynamicMetrics(alpha=500.0)`.
   - To keep the demo fast, decimate the hidden states temporally by a factor of
     50 before passing them to `calculate_ksm(window_size=5)`.
   - Interpolate the resulting KSM scores back to the original `SEQ_LEN`
     resolution using `np.interp`.

6. **The Publication-Ready Dashboard:**
   - Use `HardwareMonitor(device).run_scaling_benchmark(d_model=256)` to
     generate the VRAM scaling data.
   - Create a function `plot_bio_blade_dashboard` that generates a 3-panel
     figure (saving to `output/01_bio_blade_engine.png`) using the
     `dark_background` theme.
   - **Panel 1 (Raw Telemetry):** Subsampled 64-channel 20kHz heatmap with a
     vertical dashed line for the crash boundary. Ensure symmetric `vmin/vmax`
     based on the pre-crash 95th percentile. Decimate the time axis by 10 for
     rapid plotting. Title: "Analog Biological Layer (HD-MEA 20kHz Telemetry)".
   - **Panel 2 (Thermodynamics):** PyDMD KSM metric plotting the sudden
     biological collapse at `EVENT_FRAME`. Add a horizontal dashed red line at
     y=0.9 labeled "Stability Collapse Threshold". Title: "Digital Compute
     Layer: PyDMD Koopman Stability Metric (KSM)".
   - **Panel 3 (Hardware Telemetry):** The VRAM Hardware Monitor chart plotting
     `seq_lengths` vs `mamba_vram` (Linear) and `transformer_vram` (Quadratic).
     Title: "Hardware Invariant: Peak VRAM vs. Sequence Length".

Constraints:

- Ensure the script acts as a self-contained, high-impact demonstration. No
  references to MambaLRP or training loops (saved for later Acts).
