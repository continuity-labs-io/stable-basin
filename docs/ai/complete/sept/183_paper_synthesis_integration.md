**Context Files to Load / Modify:**
* `tools/compile_paper.py`
* `tools/llm_ghostwriter.py`
* `paper/sharpening_the_tack.tex.j2`

**Task: Phase 2.2 - Prompt Engineering and Template Integration**
We will now hook the Ghostwriter into the compiler script, design the strict system prompts to prevent hallucination, and inject the AI-generated prose into our LaTeX skeleton.

**Core Objectives:**

**1. Compile Script Integration (`tools/compile_paper.py`):**
* Import and instantiate the `LLMGhostwriter`.
* Before rendering the Jinja2 template, use the Ghostwriter to generate three distinct text blocks:
  * **Abstract:** Pass the `meta` dictionary. Prompt the LLM to write a concise, 150-word scientific abstract summarizing the core question and hypothesis.
  * **Pathology Results:** Pass the `metrics_decline` dictionary. Instruct the LLM to write a 1-paragraph summary explaining how the Gaussian model failed to detect aging, but the PrecisionWeightedEBM succeeded, specifically citing the Cohen's d and KS-statistic.
  * **Intervention Results:** Pass the `metrics_intervention` dictionary. Instruct the LLM to write a 1-paragraph summary explaining the thermodynamic rescue, specifically citing the massive spike in the Hessian trace and the intervention's Cohen's d effect size.

**2. Strict System Guardrails:**
* The `system_prompt` passed to the Ghostwriter MUST contain strict constraints. Example: *"You are a rigorous computational biology researcher writing a paper in LaTeX format. You must ONLY use the empirical numbers provided in the JSON payload. Do not hallucinate external studies, metrics, or calculations. Maintain an objective, academic tone. Do not use markdown formatting; output LaTeX-safe plaintext."*

**3. Template Re-Wiring (`paper/sharpening_the_tack.tex.j2`):**
* Pass the three generated text blocks into the Jinja2 template render context (e.g., `ai_abstract`, `ai_results_pathology`, `ai_results_intervention`).
* Update the LaTeX template to inject these variables using `\VAR{ ai_abstract }` inside the `\begin{abstract}` environment, and place the results variables in the Results section.
* Ensure they flow naturally alongside the hardcoded numbers we injected in Phase 1.

**Constraints:**
* Keep the prompt strings cleanly formatted in the Python code (e.g., using multi-line `"""` strings or extracting them to a config dict).
