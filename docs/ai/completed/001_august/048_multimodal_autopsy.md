Context: We are finalizing the "Director's Cut" of the Project MELD repository.
This task builds "Master Demo 3: The Multimodal Diagnostic", representing our
core pharmaceutical and drug-discovery pitch (e.g., for Daphne Koller, Altos
Labs, BrainStorm Therapeutics). It proves that when our continuous-time
state-space model detects a thermodynamic crash, we can deploy exact Layer-wise
Relevance Propagation (LRP-epsilon) to trace the causality backward in time,
isolating the specific biological mechanism (the drug target) that triggered the
failure, and outputting an automated, human-readable JSON diagnostic.

Task: Create a new script `src/demo/03_multimodal_diagnostic.py` by synthesizing
the 114-D simulation logic, the `MambaLRPEpsilon` module, and the
`ThermodynamicDiagnosticEngine`.

Requirements:

1. **Imports & Setup:**

   - Import `torch`, `torch.nn.functional as F`, `torch.optim`, `json`, `os`,
     `numpy`, and `matplotlib.pyplot` (using `matplotlib.use("Agg")`).
   - Import `SpikeForecaster` from `src.models.spike_forecaster`. (_Note: We use
     this instead of NeocorticalEngine because MambaLRPEpsilon explicitly
     targets its `output_proj` and `get_hidden_states` attributes_).
   - Import `MambaLRPEpsilon` from `src.metrics.mamba_lrp`.
   - Import `ThermodynamicDiagnosticEngine` from
     `src.metrics.diagnostic_engine`.
   - Import `get_optimal_device` from `src.utils.device`.
   - Setup basic console logging (INFO level).

2. **The Execution Flow (`main` function):**

   - Initialize device using `get_optimal_device(allow_mps=False, verbose=True)`
     (MPS strictly disabled to ensure accurate autograd/LRP backward pass).
   - Print a stark console header:
     `[*] BOOTING MASTER DEMO 3: THE MULTIMODAL DIAGNOSTIC`.
   - Instantiate the `SpikeForecaster` (input_dim=114, d_model=256, d_state=64).

3. **Burn-in Training (The Baseline):**

   - Generate a clean biological sequence
     `torch.randn(1, 200, 114).abs() * 0.5`.
   - Execute a rapid 15-iteration burn-in loop (AdamW, lr=1e-3).
   - Predict `T+1` and optimize using standard `F.mse_loss` on the clean data.
     (This burns the spatial covariance into the weights so LRP can accurately
     track deviations).
   - Log the training loss to the console.

4. **The Causal Crash Simulation:**

   - Clone a new validation sequence from the clean data.
   - **Inject the Root Cause (The Trigger):** At `TRIGGER_FRAME = 110`,
     artificially spike the values of the RNA panic genes. Specifically, target
     indices `101` and `102` (representing `RNA_TP53` and `RNA_CDKN2A` in our
     114-D Trifecta layout). Add `+5.0` to their amplitude for frames 110-120.
   - **Inject the Thermodynamic Crash:** At `EVENT_FRAME = 140`, simulate the
     structural collapse (apoptosis). Multiply all 114 dimensions for frames 140
     to 200 by `0.05` (the flatline).
   - _Physics Note:_ We are mathematically staging a scenario where an RNA
     stress alarm fires at T=110, causing a global network collapse at T=140.

5. **MambaLRP & The Diagnostic Engine:**

   - Put the model in `.eval()`.
   - Instantiate `MambaLRPEpsilon(engine, epsilon=1e-7)`.
   - Execute the exact relevance projection targeting the crash frame:
     `relevance_tensor = lrp.attribute(test_seq, target_time_step=EVENT_FRAME)`.
   - _Override Note:_ Explicitly monkey-patch `engine.compute_attribution` to
     use the `MambaLRPEpsilon.attribute` method so the
     `ThermodynamicDiagnosticEngine` calls the mathematically exact LRP instead
     of its naive Input\*Gradient fallback:
     `engine.compute_attribution = lambda x, t: lrp.attribute(x, t)`
   - Instantiate `ThermodynamicDiagnosticEngine(engine)`.
   - Call `generate_diagnostic(test_seq, EVENT_FRAME)`.
   - Print the resulting `diagnostic_report` beautifully formatted as a JSON
     string to the console. It MUST successfully identify the spike at T=110 on
     the `Psi` genes as the primary causal trace.

6. **The Publication-Ready Dashboard:**
   - Create
     `plot_diagnostic_dashboard(raw_seq, relevance_map, trigger_frame, event_frame)`.
     Save to `output/03_multimodal_diagnostic.png` using the `dark_background`
     theme.
   - 2-Panel Vertical Layout (figsize=12, 10, sharex=True):
     - **Panel 1 (The Biological Input):** 2D Heatmap of the corrupted
       `test_seq` (transposed). Add a vertical dotted yellow line at
       `TRIGGER_FRAME` and a dashed red line at `EVENT_FRAME`. Title: "Analog
       Biological Layer (114-D Trifecta Tensor)". Y-axis: "Features (Sigma, Psi,
       Omega)".
     - **Panel 2 (The MambaLRP Attribution):** 2D Heatmap of the `relevance_map`
       (transposed). Use a symmetric diverging colormap (e.g., `coolwarm` or
       `RdBu_r`) normalized to the 99th percentile of the absolute relevance.
       Add the same vertical lines. The heatmap MUST visually show the bright
       red cluster of predictive relevance at T=110 on indices 101/102. Title:
       "Digital Compute Layer: MambaLRPEpsilon Causal Attribution". Y-axis:
       "Rank-One Sub-Circuits".

Constraints:

- Keep the script fully self-contained.
- The console output should narrate the story (e.g.,
  `[*] Injecting RNA Stress Alarm at T=110...`,
  `[*] Structural Collapse at T=140...`,
  `[*] Generating Thermodynamic Diagnostic...`).
