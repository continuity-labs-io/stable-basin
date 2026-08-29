# TASK: Project Refactor (Stable Basin)

We will execute a workspace-wide search-and-replace to transition the project
from "MeldBenchmark" to "Stable Basin", while strictly preserving the internal
lore (e.g. `MeldEngine`, `MeldLoss`).

**WARNING on Root Directory Rename:** Renaming the active workspace folder
(`MeldBenchmark` -> `stable-basin`) from within an active IDE session can break
the editor's path bindings. Handle the top-level directory rename manually in
the OS finder _after_ the code is updated.

## Proposed Changes

### 1. Configuration & Root Files

- **`pyproject.toml`**:
  - Rename `name = "meld-benchmark"` to `name = "stable-basin"`
  - Update description to `"Stable Basin"`
- **`README.md`**:
  - Replace `# MeldBenchmark` with `# Stable Basin`
  - Add the required philosophy: _"The objective is to maintain the biological
    latent state inside the youthful homeostatic attractor basin, evaluated by
    Time-in-Basin (TiB) against entropic decay and simulated hardware
    failures."_
- **`ISSUES.md`**:
  - Replace "MeldBenchmark" with "Stable Basin Benchmark"
- **`LICENSE`**:
  - Replace "MeldBenchmark Contributors" with "Stable Basin Contributors"

### 2. Documentation Updates

Search and replace `MeldBenchmark` -> `Stable Basin Benchmark` in the following
files:

- `docs/ai/rules/001_precision_standards.md`
- `docs/ai/rules/002_logging_standards.md`
- `docs/ai/completed/000_july/023_consolidate_the_core.md`
- `docs/ai/completed/001_august/032_continuous_hd_mea_dataset.md`
- `docs/ai/completed/000_july/011_spd_mamba_integration.md`

### 3. Internal Code (Exclusions)

- All python code (e.g. `meld_loss.py`, `MeldEngine`) must remain completely
  **untouched** per the internal Lore preservation rule.

## Verification

1. Run `git diff` to ensure no python files containing `meld` were accidentally
   modified.
2. Run `grep -ri "MeldBenchmark"` to ensure all public-facing occurrences are
   eradicated.
