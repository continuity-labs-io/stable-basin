# Extending the Paper Generation Pipeline

The user has requested that we expand the `sharpening_the_tack` paper generation pipeline to include the detailed methodologies discussed and the new baseline metrics we just computed. 

## Proposed Changes

### 1. `paper/sharpening_the_tack/paper_metadata.yaml`
We will update the YAML metadata to feed the LLM Ghostwriter the necessary data and prompts to write the new sections:

**Data Sources:**
- Add `metrics_baseline: "output/echo/benchmarks/08_worm_gait_baseline_metrics.json"` to the `data_sources` list.

**Ghostwriter Sections:**
- **[NEW] `ai_methods`**: 
  - **Prompt**: "Write a highly detailed 'Methods' section explaining the architecture of the physics engine. You must detail the 46-dimensional state space partitioned into a Micro and Macro observer via Fristonian Markov Blankets. You must also detail the neural network architecture used to learn the energy landscape: the `PrecisionWeightedEBM`, which consists of a shared MLP backbone with GELU activations (to ensure smooth, twice-differentiable physics), splitting into a scalar Energy head and a Precision head that enforces a strictly Symmetric Positive-Definite matrix via Cholesky decomposition."
  - **Context**: `config_intervention` (This contains the observer dimensions and network depths/widths).
- **[NEW] `ai_results_baseline`**:
  - **Prompt**: "Write the opening to the 'Results' section summarizing the baseline biological metrics. Using the provided metrics, detail the differences in the Time Domain (Critical Slowing Down), Spectral Domain, and Entropy Production between the Young and Old C. elegans worms. Crucially, you must explicitly explain to the reader that a higher Hessian trace is 'better' because it indicates a steep, robust thermodynamic basin that resists noise, whereas a low trace indicates a flattened, pathological landscape."
  - **Context**: `metrics_baseline`.

### 2. `paper/sharpening_the_tack/sharpening_the_tack.tex.j2`
We will update the Jinja2 LaTeX template to inject the new AI-generated sections into the document.

**Modifications:**
- In `\section{Methods}`, replace the stubbed sentence with `\VAR{ ai_methods }`.
- In `\section{Results}`, inject `\VAR{ ai_results_baseline }` at the very beginning of the section, right before the pathology and intervention discussions.

## Verification
- Run `make paper` to trigger the ghostwriter pipeline.
- Verify that `paper/sharpening_the_tack/sharpening_the_tack.tex` successfully compiles and contains the gory architectural details in the Methods and the baseline metrics (with the trace explanation) in the Results.
