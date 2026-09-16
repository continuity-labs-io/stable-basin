Context: We need to implement the Tier 1 VRAM Hardware Dashboard for Project
CHRONOS. We will modularize the logic from `src/demo/cell_observatory.py` into a
reusable metric class and inject it as the 4th panel in our `e2e_demo.py`
dashboard to visually prove Mamba's linear memory footprint against a
Transformer's quadratic explosion.

Task 1: Create `src/metrics/hardware_monitor.py` Create a PyTorch utility class
`HardwareMonitor`.

- Import `torch`, `torch.nn`, and `Mamba` from `mamba_ssm`.
- `__init__(self, device)`: Store the device.
- `run_scaling_benchmark(self, d_model=768, seq_lengths=[100, 500, 1000, 2000, 4000, 8000])`:
  - If `self.device.type != 'cuda'`, print a warning and return mock lists for
    CPU demonstration (e.g., linear vs quadratic math values to keep the demo
    from failing on laptops).
  - Instantiate a standard Transformer
    (`torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True).to(self.device)`)
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
      `torch.cuda.empty_cache()`, and append `None` or a massive spike value to
      simulate the crash for the remaining lengths.
  - Return `seq_lengths`, `mamba_vram`, `transformer_vram`.

Task 2: Update `src/demo/e2e_demo.py`

- Import `HardwareMonitor` from `src.metrics.hardware_monitor`.
- After calculating the Hysteresis metric (and before plotting), instantiate
  `hw_monitor = HardwareMonitor(device)`.
- Determine the correct `d_model` dimension from `latent_sequence.shape[-1]` (to
  automatically handle whether the GEVI fusion is active or not).
- Run the benchmark:
  `seq_lengths, mamba_vram, transformer_vram = hw_monitor.run_scaling_benchmark(d_model=d_model)`.
- Update the Matplotlib figure to a 4-panel layout:
  `fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))`.
- Add Plot 4 (`ax4` - The Hardware Telemetry):
  - Plot `seq_lengths` vs `mamba_vram` (Color: 'lime', Line: solid, Label:
    'CHRONOS Mamba-2 (Linear $O(N)$)').
  - Plot `seq_lengths` vs `transformer_vram` (Color: 'red', Line: dashed, Label:
    'Legacy Transformer (Quadratic $O(N^2)$)').
  - Title: "Hardware Invariant: Peak VRAM vs. Sequence Length", color='white',
    fontweight='bold'.
  - Y-label: "Peak VRAM Allocated (MB)". X-label: "Continuous Time Context
    (Frames)".
  - Add a horizontal dotted line representing a typical "24GB Edge GPU Limit"
    (y=24000) to show exactly where the Transformer crosses the death line while
    Mamba stays flat.
- Update `plt.tight_layout()` and save to `plot_dashboard_final.png`.
