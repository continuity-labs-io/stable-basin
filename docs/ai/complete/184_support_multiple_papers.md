# Support Multiple Papers (Prompt 184)

**Context:** The current paper compilation pipeline (`tools/compile_paper.py`) and `Makefile` are hardcoded for a single paper (`sharpening_the_tack`). To future-proof the repository and support multiple papers, we need to transition to a generic, data-driven architecture.

**Core Objectives:**

1. **Directory Restructuring:**
   - Move all current paper assets (`paper_metadata.yaml`, `sharpening_the_tack.tex.j2`, and build artifacts) from `paper/` into a dedicated subfolder: `paper/sharpening_the_tack/`.

2. **Data-Driven Compilation (`tools/compile_paper.py`):**
   - Update the script to accept a `--paper` CLI argument (defaulting to `sharpening_the_tack`).
   - Extract hardcoded data sources (e.g., `configs/worm_gait_intervention.yaml`, `output/echo/benchmarks/06_worm_gait_metrics.json`) and ghostwriter prompts into the paper's `paper_metadata.yaml` file.
   - Refactor the compiler to dynamically load whatever data sources and ghostwriter sections are defined in the metadata, making the python script entirely paper-agnostic.
   - Update Jinja2 environment and file paths to read templates from and write output to `paper/{paper_name}/`.

3. **Makefile Updates:**
   - Modify the `paper` target to accept a `PAPER` parameter (e.g., `make paper PAPER=sharpening_the_tack`).
   - Ensure it calls the python script with the `--paper` flag and runs `pdflatex` in the correct subfolder.

4. **Configuration Extraction (`paper/sharpening_the_tack/paper_metadata.yaml`):**
   - Extend the YAML file to include a `data_sources` mapping (pointing to the JSON/YAML files) and a `ghostwriter_sections` mapping (defining the specific prompts and context variables for the abstract and results).

**Constraints & Verification:**
- Running `make paper` without arguments must still correctly compile `sharpening_the_tack` end-to-end identically to the current implementation.
