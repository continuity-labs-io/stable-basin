**Context Files to Load / Create:**
* `docs/EXECUTABLE_PAPER_DESIGN.md` (Read to understand the pipeline architecture)
* `paper/paper_metadata.yaml` (Create in paper/ directory)
* `paper/sharpening_the_tack.tex.j2` (Create in a new `paper/` directory)

**Task: Phase 1.1 - Create the Executable Paper Skeleton**
We are beginning Phase 1 of the Executable Paper CI/CD pipeline. The goal of this ticket is to create the human "Axiom" input file and a Jinja2-templated LaTeX skeleton that will eventually receive our empirical data.

**Core Objectives:**

**1. Create the Axiom File (`paper/paper_metadata.yaml`):**
* In the `paper/` directory, create a simple YAML file containing the high-level philosophical constraints of the paper.
* Include these exact keys and values:
  * `title`: "Sharpening the Tack: Thermodynamic Intervention in Waddington Basins"
  * `author`: "Stable Basin Project"
  * `core_question`: "Can we mathematically detect and computationally reverse biological aging using continuous-time State Space Models?"
  * `hypothesis`: "Aging is a measurable geometric flattening of the macroscopic Waddington basin. Injecting exogenous precision will artificially steepen this basin and rescue failing biological limit cycles."

**2. Create the Jinja2 LaTeX Template (`paper/sharpening_the_tack.tex.j2`):**
* Create a new directory named `paper/` in the repo root.
* Inside it, create a valid LaTeX document skeleton using the `article` class. Include standard packages (e.g., `geometry`, `graphicx`, `amsmath`).
* **LaTeX-Safe Jinja2 Tags:** Because standard Jinja2 `{{ }}` tags conflict with LaTeX, we will use `\VAR{ variable_name }` for variable injection.
* Inject the placeholders where data will eventually flow:
  * The Title and Author block should pull from `\VAR{ meta.title }` and `\VAR{ meta.author }`.
  * In the Introduction section, include a placeholder for the hypothesis: `\VAR{ meta.hypothesis }`.
  * In the Results section, write a placeholder sentence describing the Wormgate benchmark results. It must reference the exact nested JSON keys from our benchmarks. For example:
    "The baseline Gaussian model yielded a KS-statistic of \VAR{ metrics_decline.GaussianEBM.ks_statistic | round(4) }. However, the Precision Weighted EBM detected significant flattening with a Cohen's d of \VAR{ metrics_decline.PrecisionWeightedEBM.cohens_d | round(4) }."
  * Add a similar sentence referencing the intervention rescue metrics:
    "Applying an intervention of $\lambda = \VAR{ config_intervention.lambda_B }$ steepened the basin to a trace of \VAR{ metrics_intervention.RunB.overall_mean | round(2) }, yielding a rescue effect size of $d = \VAR{ metrics_intervention.Comparisons.cohens_d \vert{} round(2) }$."

**Constraints:**
* Keep the LaTeX document short and structural (just an Abstract, Intro, Methods, and Results section). We are only testing the plumbing right now.
* Ensure the Jinja2 variable names cleanly match the structure of our existing YAML and JSON files.
