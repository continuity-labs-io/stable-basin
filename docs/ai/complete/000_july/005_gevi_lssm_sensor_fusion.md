Context: We are executing Tier 1 of the MELD Sprint Backlog. We need to
implement two major components to finalize our "Proof of Life" pipeline:

1. The "GEVI + LLSM Sensor Fusion Injector" to demonstrate multi-rate sensor
   fusion and its impact on thermodynamic metrics (CSD and KSM).
2. The "VRAM Hardware Dashboard" to visually prove Mamba's linear memory
   footprint against a Transformer's quadratic explosion. No new dependencies
   are required. Use the existing `src` directory structure.

Task 1: Create `src/models/gevi_injector.py` Create a PyTorch module
`GEVIInjector(nn.Module)` that simulates a high-frequency (20kHz) bioelectric
data stream and temporally pools it to match a 100Hz optical framerate.

- `__init__(self, gevi_sample_rate=20000, target_clock_hz=100, gevi_dim=64)`:
  - Calculate `self.compression_ratio = int(gevi_sample_rate / target_clock_hz)`
    (should evaluate to 200).
  - Define
    `self.compressor = nn.Conv1d(in_channels=1, out_channels=gevi_dim, kernel_size=self.compression_ratio, stride=self.compression_ratio)`.
- `generate_synthetic_gevi(self, batch_size, target_time_steps, device, is_healthy=True)`:
  - `total_steps = target_time_steps * self.compression_ratio`.
  - Create a baseline tensor at -70.0 mV of shape `(batch_size, 1, total_steps)`
    on the target `device`.
  - Add normal thermal noise (e.g., `torch.randn(...) * 2.0`).
  - Add sparse action potential spikes (+100.0 mV) to ~1% of steps randomly.
  - If `not is_healthy`: Inject a "variance explosion" (std=40.0) starting from
    the equivalent of Frame 6 (index `6 * self.compression_ratio`) to the end of
    the sequence. This simulates early-warning voltage jitter _before_ the
    structural cell rupture occurs at Frame 7.
  - Return the tensor.
- `forward(self, batch_size, target_time_steps, device, is_healthy=True)`:
  - Call `self.generate_synthetic_gevi(...)`.
  - Pass the result through `self.compressor`.
  - Transpose the output from `(Batch, Channels, Time)` to
    `(Batch, Time, Channels)`.
  - Return the compressed latent representation of shape
    `(Batch, Time, gevi_dim)`.

Task 2: Create `src/metrics/hardware_monitor.py` Create a PyTorch utility class
`HardwareMonitor`.

- Import `torch`, `torch.nn`, and try to import `Mamba` from `mamba_ssm`.
- `__init__(self, device)`: Store the device.
- `run_scaling_benchmark(self, d_model=832, seq_lengths=[100, 500, 1000, 2000, 4000, 8000])`:
  - If `self.device.type != 'cuda'`, print a warning and return mock lists for
    CPU demonstration (e.g.,
    `mamba_vram = [15.0 + (L * d_model * 4 * 16 / (1024**2)) for L in seq_lengths]`
    and
    `transformer_vram = [15.0 + (L * d_model * 4 / (1024**2)) + (L**2 * 8 * 4 / (1024**2)) for L in seq_lengths]`).
  - Otherwise, instantiate a standard Transformer
    (`nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True).to(self.device)`)
    and a Mamba block
    (`Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to(self.device)`).
  - Initialize empty lists: `mamba_vram = []` and `transformer_vram = []`.
  - For `L` in `seq_lengths`:
    - Generate a dummy tensor `x = torch.randn(1, L, d_model).to(self.device)`.
    - **Test Mamba:** Call `torch.cuda.reset_peak_memory_stats()`. Run
      `with torch.no_grad(): _ = mamba(x)`. Append
      `torch.cuda.max_memory_allocated() / (1024**2)` to `mamba_vram`.
    - **Test Transformer:** Call `torch.cuda.reset_peak_memory_stats()`. Run
      `with torch.no_grad(): _ = attn(x, x, x)`. Append
      `torch.cuda.max_memory_allocated() / (1024**2)` to `transformer_vram`. If
      it throws an `OutOfMemoryError` or `RuntimeError`, catch it, call
      `torch.cuda.empty_cache()`, and append `None` (or a massive spike value)
      for the remaining lengths.
  - Return `seq_lengths`, `mamba_vram`, `transformer_vram`.

Task 3: Update `src/demo/e2e_demo.py` Integrate the GEVI injector and Hardware
Monitor, and update the dashboard to plot the comparisons.

- **Imports:** Import `GEVIInjector` from `src.models.gevi_injector` and
  `HardwareMonitor` from `src.metrics.hardware_monitor`.
- **Instantiation:** After initializing `mamba_engine` (d_model=768),
  initialize:
  - `gevi_injector = GEVIInjector().to(device)`
  - `mamba_engine_fused = VectorSeqEngine(d_model=768 + 64).to(device)`
  - Set both to `.eval()`.
- **Latent Fusion:** After extracting `latent_anomalous` and `latent_healthy`:
  - Generate GEVI latents:
    `gevi_anomalous = gevi_injector(experimental_batch.size(0), experimental_batch.size(1), device, is_healthy=False)`
    `gevi_healthy = gevi_injector(raw_batch.size(0), raw_batch.size(1), device, is_healthy=True)`
  - Fuse them:
    `latent_fused_anomalous = torch.cat([latent_anomalous, gevi_anomalous], dim=-1)`
    and
    `latent_fused_healthy = torch.cat([latent_healthy, gevi_healthy], dim=-1)`.
- **Processing:**
  - Pass `latent_anomalous` through `mamba_engine` -> extract loss and ksm
    metric.
  - Pass `latent_fused_anomalous` through `mamba_engine_fused` -> extract loss
    and ksm metric.
  - Calculate CSD and KSM using `ThermodynamicMetrics` for BOTH
    `latent_anomalous[0].detach()` (Optics-only) AND
    `latent_fused_anomalous[0].detach()` (Fused).
  - Update the Hysteresis calculation to use the fused representations
    (`latent_fused_healthy` and the rescue trajectory generated by
    `mamba_engine_fused` starting from `latent_fused_anomalous[:, 7:8, :]`).
- **Hardware Telemetry:** Run `hw_monitor = HardwareMonitor(device)` and
  `seq_lengths, mamba_vram, transformer_vram = hw_monitor.run_scaling_benchmark(d_model=832)`.
- **Plotting (Update `plot_metrics.png`):**
  - Change subplot layout to:
    `fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))`.
  - **ax1 (KSM) & ax2 (CSD):** Plot TWO lines per metric: Optics-Only
    (Cyan/Magenta for KSM/CSD, marker='o', linestyle='--', label='Optical-Only')
    and Fused (Yellow/Orange for KSM/CSD, marker='s', linewidth=3, label='Fused
    Multi-Modal').
  - For `ax1` and `ax2`: Ensure the vertical lines still denote "Injected
    Structural Anomaly (T6->T7)". Add a new vertical line (`ax.axvline`) and
    span (`ax.axvspan`) at `x=6` (Color e.g., 'gold') for "Bioelectric Variance
    Explosion (T5->T6)".
  - **ax3 (Hysteresis):** Plot the Path Divergence for the Fused path.
  - **ax4 (VRAM Dashboard):**
    - Plot `seq_lengths` vs `mamba_vram` (Color: 'lime', Line: solid, Label:
      'MELD Mamba-2 (Linear $O(N)$)').
    - Plot `seq_lengths` vs `transformer_vram` (Color: 'red', Line: dashed,
      Label: 'Legacy Transformer (Quadratic $O(N^2)$)').
    - Title: "Hardware Invariant: Peak VRAM vs. Sequence Length", Y-label: "Peak
      VRAM Allocated (MB)", X-label: "Continuous Time Context (Frames)".
    - Add a horizontal dotted line at `y=24000` labeled "24GB Edge GPU Limit".
