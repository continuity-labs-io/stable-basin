# Design Document: CI/CD for Academic Publishing ("The Executable Paper")

## 1. Executive Summary

This document outlines the architecture for a Continuous Integration/Continuous Deployment (CI/CD) pipeline dedicated to academic publishing within the *Stable Basin* repository. The traditional academic model—where a scientific paper is a static, hand-written document entirely decoupled from the codebase that generated its claims—is brittle, non-reproducible, and highly vulnerable to human error during revisions.

We propose a paradigm shift: treating the scientific paper as a **compilable software artifact**. By integrating templating engines (e.g., Jinja2, Quarto) and LLM-driven synthesis directly into the repository, the entire scientific pipeline—from experiment simulation to final PDF generation—becomes a deterministic, push-button process.

## 2. Core Philosophy & Value Proposition

If the scientific method is formalized as `Hypothesis -> Model -> Experiment -> Results`, then the resulting paper is simply the "UI" for the codebase.

*   **Immutable Ground Truth:** The codebase (`configs/`, `src/`) is the single source of truth. The paper is merely a downstream projection of that code.
*   **The "Reviewer 2" Defense:** When peer review demands parameter changes (e.g., "Change the baseline precision gain $\lambda$ from 0.2 to 2.5"), the researcher simply updates the YAML configuration and types `make paper`. The entire pipeline re-runs, the metrics update, the LLM agent re-writes the interpretation based on new data, and a new PDF is compiled.
*   **Forkable Science:** Anyone can fork the repository, swap the input dataset or tweak the physics parameters, and compile a structurally identical, peer-review-ready paper on a new biological modality.

## 3. Pipeline Architecture

The pipeline consists of four distinct stages, triggered sequentially via a single `Makefile` command (`make paper`).

### Stage 1: The "Axiom" File (Human Input)
The human researcher acts as the architect, defining only the high-level intent and philosophical constraints of the paper in a single `paper_metadata.yaml` file. No manual drafting of methodology or results is performed.

```yaml
# paper_metadata.yaml
title: "Sharpening the Tack: Thermodynamic Intervention in Waddington Basins"
core_question: "Can we detect and reverse biological aging using continuous-time State Space Models?"
hypothesis: "Top-down precision injection will steepen the Effective Hessian Trace and rescue failing biological limit cycles."
```

### Stage 2: Empirical Generation (The Physics Engine)
The core physics engine (e.g., Stable Basin JAX/Equinox models) executes the experiments defined in the standard configuration files (e.g., `configs/worm_gait_intervention.yaml`).

* **Outputs:** High-fidelity `.png` figures and strict JSON serialization of all quantitative results (`metrics.json`). This ensures no data is trapped in stdout or volatile memory.

### Stage 3: The Agentic Synthesizer (`tools/compile_paper.py`)

A Python script orchestrates the synthesis of the paper. It ingests the Axiom file, the YAML configurations, and the `metrics.json` outputs.

* **LLM Integration:** The script passes these highly structured inputs to an LLM API (e.g., Claude, GPT-4) with strict system prompts.
* **Prompt Example:** "You are an academic writer. Using ONLY the parameters in the YAML and the empirical results in the JSON, write a 3-paragraph Discussion section explaining how the strong intervention ($\lambda=5.0$) successfully restored thermodynamic homeostasis, referencing the Hessian Trace spiking from 18 to 443."
* **Output:** The LLM generates the interpretative text blocks (Abstract, Introduction, Discussion) and saves them as markdown or raw text snippets.

### Stage 4: The Compiler (Jinja2 -> LaTeX/Quarto -> PDF)

A skeletal LaTeX or Markdown template (`main.tex.j2` or `paper.qmd`) contains placeholders for both the raw data and the AI-generated text. The script uses a templating engine (like Jinja2) to aggressively inject the variables and compile the final document.

```latex
% Example Jinja2/LaTeX Template Snippet
\section{Methods}
The simulation was run for {{ config.experiment.N_steps }} steps with a discretization of $dt = {{ config.experiment.dt }}$.

\section{Results}
The intervention increased the Effective Hessian Trace from {{ metrics.RunA.overall_mean | round(2) }} to {{ metrics.RunB.overall_mean | round(2) }} (Cohen's $d = {{ metrics.Comparisons.cohens_d | round(2) }}$).

\input{generated/ai_discussion.tex}
```

## 4. Implementation Plan
The architecture will be implemented iteratively to ensure the core physics engine remains isolated and mathematically sound during development.

### Phase 1: Templating & Data Injection (Future Sprint)
* Establish the `paper_metadata.yaml` standard.
* Create a basic Jinja2 LaTeX template for the "Sharpening the Tack" paper.
* Write a naive `compile_paper.py` that successfully reads `metrics.json` and `configs/` and injects those raw numbers into the LaTeX template.

### Phase 2: Agentic Synthesis Integration
* Integrate an LLM API client into `compile_paper.py`.
* Develop the strict system prompts required to constrain the LLM from hallucinating data outside of the provided JSON/YAML boundaries.
* Implement a caching layer so the LLM is only queried if the underlying `metrics.json` or `paper_metadata.yaml` has changed (saving API costs and ensuring idempotent builds).

### Phase 3: Full Automation
* Add the `.PHONY: paper` target to the global `Makefile`.
* Integrate the compilation step into the GitHub Actions CI/CD pipeline, automatically generating a "Latest Paper PDF" artifact on every push to the `main` branch.

## 5. Security and Hallucination Mitigation
The primary risk of integrating LLMs into academic publishing is hallucination. This pipeline mitigates this risk through Structural Constraint:

* The LLM never touches the raw data arrays or performs calculations.
* All statistics (p-values, effect sizes, means) are calculated deterministically by SciPy/Pingouin in Stage 2 and injected directly by Jinja2 in Stage 4.
* The LLM is strictly confined to generating the semantic wrappers (the prose) around the hardcoded variables and evaluating the hypothesis given the extracted facts.
