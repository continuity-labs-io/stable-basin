# Plan: Include Experimental Plots in the Paper

To elevate the paper to biophysics journal standards, we will formally embed the generated experimental plots directly into the compiled PDF, complete with rigorous scientific captions.

## Proposed Changes

### 1. Relocate Plot Assets
LaTeX requires image assets to be accessible during the compilation phase. We will copy the two plots:
- `output/echo/benchmarks/06_worm_gait_decline_ablation.png`
- `output/echo/benchmarks/07_worm_gait_intervention_rescue.png`
into the `paper/sharpening_the_tack/` directory.

### 2. Generate Formal Captions (`paper_metadata.yaml`)
Instead of hardcoding simple captions, we will extend the AI Ghostwriter pipeline to generate formal, rigorous journal-style captions. We will add two new sections to `paper_metadata.yaml`:
- **`ai_caption_decline`**:
  - Prompt: "Write a formal 2-sentence figure caption for the 'Worm Gait Decline Ablation Plot'. Explain that the figure demonstrates the failure of the baseline Gaussian model and the successful detection of the flattened Waddington basin by the PrecisionWeightedEBM, citing the relevant KS-statistics. Do not output anything other than the caption text."
  - Context: `metrics_decline`
- **`ai_caption_rescue`**:
  - Prompt: "Write a formal 2-sentence figure caption for the 'Thermodynamic Intervention Rescue Plot'. Explain that the figure visualizes the steepening of the thermodynamic basin and the rescue of the limit cycle after the precision injection, citing the massive spike in the Hessian trace. Do not output anything other than the caption text."
  - Context: `metrics_intervention`

### 3. Update LaTeX Template (`sharpening_the_tack.tex.j2`)
We will embed standard `\begin{figure}` blocks into the Results section of the template:
- After `\VAR{ ai_results_pathology }`, inject Figure 1 including `\includegraphics{06_worm_gait_decline_ablation.png}` with `\caption{\textbf{Figure 1. Worm Gait Decline Ablation Plot.} \VAR{ ai_caption_decline }}`.
- After `\VAR{ ai_results_intervention }`, inject Figure 2 including `\includegraphics{07_worm_gait_intervention_rescue.png}` with `\caption{\textbf{Figure 2. Thermodynamic Intervention Rescue Plot.} \VAR{ ai_caption_rescue }}`.

## Verification
- We will execute the data copy and update the configurations.
- We will run `make paper` to trigger the ghostwriter pipeline to write the captions and compile the PDF.
- We will verify that the images render correctly in the output PDF.
