**Context Files to Load / Create:**
* `paper_metadata.yaml`
* `paper/sharpening_the_tack.tex.j2`
* `configs/worm_gait_intervention.yaml`
* `output/echo/benchmarks/06_worm_gait_metrics.json`
* `output/echo/benchmarks/07_worm_gait_intervention_metrics.json`
* `tools/compile_paper.py` (Create)
* `Makefile` (Modify)

**Task: Phase 1.2 - The Deterministic Paper Injector**
We need to build the Python script that reads our physical outputs and configuration files, injects them into the Jinja2 LaTeX template, and compiles the final PDF.

**Core Objectives:**

**1. Build the Compilation Script (`tools/compile_paper.py`):**
* Create the script using the standard `json`, `yaml`, and `jinja2` libraries.
* The script must load the following data sources into memory as Python dictionaries:
  * `meta`: from `paper_metadata.yaml`
  * `config_intervention`: from `configs/worm_gait_intervention.yaml`
  * `metrics_decline`: from `output/echo/benchmarks/06_worm_gait_metrics.json`
  * `metrics_intervention`: from `output/echo/benchmarks/07_worm_gait_intervention_metrics.json`
* Handle potential `FileNotFoundError` exceptions gracefully, logging a warning if a metrics file hasn't been generated yet, but allowing the script to proceed with empty dictionaries to prevent fatal crashes during setup.
* **LaTeX-Safe Jinja2 Environment:** Set up a Jinja2 Environment to load `paper/sharpening_the_tack.tex.j2`. You MUST override the default Jinja2 delimiters so they don't break LaTeX. Use the following kwargs in `jinja2.Environment`:
  * `block_start_string='\BLOCK{'`, `block_end_string='}'`
  * `variable_start_string='\VAR{'`, `variable_end_string='}'`
  * `comment_start_string='\#{'`, `comment_end_string='}'`
* Render the template passing in the four data dictionaries.
* Write the rendered output to `paper/sharpening_the_tack.tex`.

**2. Update the `Makefile`:**
* Add a new `.PHONY: paper` target.
* The command should execute sequentially:
  1. `python tools/compile_paper.py`
  2. `cd paper && pdflatex sharpening_the_tack.tex`
* *Note: Assume the host machine has `pdflatex` installed.*

**Constraints:**
* Do NOT import or implement any LLM/AI APIs in this script yet. This is strictly deterministic plumbing.
* Use the standard Python `logging` module to output clear, professional progress messages (e.g., `logger.info("Successfully injected metrics into LaTeX template.")`).
