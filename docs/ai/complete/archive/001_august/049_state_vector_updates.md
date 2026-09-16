Context: We are planting a flag for the future of the MELD architecture. We need
to formalize the "129-D Quintet Tensor" (adding Epigenetic Gamma and Metabolomic
Mu modalities) and the "ATP Metabolic Checkbook" (a hard physics constraint on
biological free energy) into the repository as a prototype and public roadmap.

Task 1: Create a prototype demo script
`src/demo/04_future_quintet_simulation.py`. Requirements:

1. **Setup:** Import `torch`, `os`, `matplotlib.pyplot` (using `Agg`), and
   `NeocorticalEngine` (from `src.models.neocortical_engine`).
2. **The 129-D Engine:** In `main()`, initialize a 129-D `NeocorticalEngine`
   (d_model=256, d_state=64). Print console statements narrating the 5
   modalities: Sigma (100D, Morphology), Psi (12D, RNA), Omega (2D, Voltage),
   Gamma (10D, Epigenetic Drift - highly sparse), and Mu (5D, Metabolomic Flux -
   including ATP).
3. **The Metabolic Checkbook Simulation:** Generate a mock continuous sequence
   of shape `[1, 500, 129]`. Isolate the final dimension (index 128) as the ATP
   reserve. Linearly deplete the ATP reserve from 1.0 down to 0.0 at frame 300.
4. **The Causal Crash:** When ATP hits 0.0 at frame 300, simulate a cascading
   Waddington crash by dropping the Voltage (Omega) and structural (Sigma)
   dimensions to near-zero variance.
5. **The Hardware Proof:** Pass the tensor through the `NeocorticalEngine`
   forward pass to prove the Mamba-2 backbone natively absorbs the 129-D
   multi-rate tensor without crashing.
6. **The Dashboard:** Save a 2D heatmap of the 129-D tensor to
   `output/04_future_quintet_simulation.png`. Add a horizontal green dashed line
   at index 128 labeled "ATP Metabolic Reserve" and a vertical red dashed line
   at frame 300 labeled "ATP Exhaustion (Waddington Crash)".

Task 2: Append these architectural goals to `ISSUES.md`. Requirements:

1. Add a new section `## 3. V2 Architecture: The 129-D Quintet Tensor`. Detail
   the need for data engineers to build actual dataloaders for single-cell
   epigenetic clocks (Gamma) and in-line electrochemical sensors (Mu). Explain
   how Mamba-2's data-dependent step size ($\Delta t$) will gracefully absorb
   the massive `NaN` gaps of hourly epigenetic reads.
2. Add a new section `## 4. The ATP Metabolic Checkbook Loss Constraint`.
   Describe the need to update `src/models/meld_loss.py`. Define the physics
   constraint: If the model predicts high-frequency action potentials or massive
   RNA transcription, it must mathematically subtract from the global ATP
   reserve dimension. If the model hallucinates a high-energy repair cascade
   while the ATP checkbook is empty, the loss function must geometrically
   explode.
