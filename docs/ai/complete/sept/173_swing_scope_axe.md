**Context Files to Load / Modify:**

- `docs/scope.md` (Read to understand the boundaries)
- `src/echo/benchmarks/` (Scan directory)
- `src/data/` (Scan directory)

**Task: Enforce The Scope Axe & Populate the Icebox** The repository has drifted
past our defined Complexity Ceiling. We must ruthlessly prune distracting,
high-dimensional, or incomplete experiments to refocus all engineering attention
on verifying the thermodynamic math of the Worm Gait limit cycle.

**Core Objectives:**

**1. Icebox Infrastructure:**

- Ensure a root-level `icebox/` directory exists.
- Create mirrored subdirectories inside it: `icebox/src/data/eeg/`,
  `icebox/src/echo/benchmarks/`.

**2. The Benchmark Purge:**

- Scan `src/echo/benchmarks/`.
- Identify any scripts related to Phase 4 (Human EEG, LEMON, Sleep-EDF) or
  non-linear EEG entropy classifiers. (e.g., `07_nonlinear_eeg_entropy.py`, or
  similar exploratory scripts).
- Move these complex, out-of-scope benchmark scripts into
  `icebox/src/echo/benchmarks/`.
- **Keep strictly:** The MOU linear ground truth tests,
  `06_worm_gait_decline.py`, and `07_insilico_reprogramming.py`.

**3. The Data Loader Purge:**

- Scan `src/data/` and `src/data/eeg/`.
- Identify any MNE-based data loaders or complex biological parsers (e.g.,
  `sleep_edf.py`, `lemon.py`).
- Move these high-dimensional data loaders into `icebox/src/data/eeg/`.
- **Keep strictly:** `celegans_gait_dataset.py` (and any synthetic tools
  required for tests).

**4. Sanitization:**

- Remove any dead imports in the core `__init__.py` files caused by these moves.
- Use `logger.info` to output a clean terminal summary detailing exactly which
  files were banished to the icebox to maintain the CFD engine focus.

**Constraints:**

- Use standard filesystem moves (`shutil.move` or git commands). Do not
  permanently delete the files; safely isolate them in the `icebox/` tree for
  future phases.
- Do not modify the Equinox physics engine or the Worm Gait benchmark files
  during this operation.
