Create a new Python script named `src/demo/7_neocortical_benchmark_demo.py` that
serves as the master integration demo for the Neocortical Assembloid Benchmark.

Context: This script ties together the `NeocorticalAssembloidDataset` and the
`NeocorticalEngine` (Mamba-2 backbone), trains the network using
physics-informed `MeldLoss`, injects a metabolic tissue crash mid-stream, and
runs MambaLRP attribution to map root-cause failure points.

Requirements:

1. Setup & Imports:
   - Import `torch`, `torch.optim`, `numpy`, `matplotlib.pyplot`, and
     `matplotlib` (using `matplotlib.use("Agg")` to prevent UI blocking).
   - Import `NeocorticalAssembloidDataset` from
     `src.pipeline.neocortical_assembloid_dataloader`.
   - Import `NeocorticalEngine` from `src.models.neocortical_engine`.
   - Import `MeldLoss` from `src.models.meld_loss`.
   - Import `get_optimal_device` from `src.utils.device`.
2. Execution Flow (`main` function):
   - Initialize the optimal device (defaulting `allow_mps=False` for safety
     during Mamba training).
   - Instantiate the `NeocorticalAssembloidDataset` (time_steps=200,
     num_channels=114) and wrap it in a PyTorch `DataLoader`.
   - Instantiate the `NeocorticalEngine` and the physics-constrained `MeldLoss`
     (alpha=1.0, beta=0.1, gamma=0.5, L=1.5).
   - Run a rapid burn-in training loop (e.g., 20 iterations) using `AdamW` to
     optimize the engine against the continuous multi-modal tensor stream.
3. Simulation of the Waddington Crash & Attribution:
   - Pull a validation sequence from the dataset. At `EVENT_FRAME = 140`, inject
     a severe metabolic drop (multiply subsequent frames by 0.05).
   - Run a forward pass to extract hidden states and compute the first-order
     Taylor attribution map (Input \* Gradient) targeting `EVENT_FRAME`.
4. Visualization Dashboard:
   - Generate a publication-ready, 2-panel Matplotlib figure with a dark
     background theme and save it to
     `output/7_neocortical_benchmark_dashboard.png`.
   - Panel 1 (Top): Heatmap of the Multi-Modal Biological Input across time and
     feature dimensions, highlighting the `EVENT_FRAME` crash boundary.
   - Panel 2 (Bottom): Heatmap of the MambaLRP Feature Attribution, proving the
     model successfully isolates the upstream sub-circuits driving the crash.
5. Execution Guard:
   - Include a standard `if __name__ == "__main__": main()` block to run the
     script directly from the terminal.
