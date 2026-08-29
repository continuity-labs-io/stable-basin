Please execute the following architectural cleanup to icebox our deterministic
Mamba and LRP components. We are pivoting to hierarchical, energy-based state
space models (using Directed Factor Graphs) and abandoning the need for custom
GPU kernels and deterministic workarounds.

1. **Create an Icebox:** Create a new directory at `src/icebox/`.
2. **Move Files:** Move the following files into `src/icebox/`:
   - `src/models/ssm/masr_mamba.py`
   - `src/models/ssm/triton_fused_scan.py`
   - `src/metrics/mamba_lrp.py`
3. **Update Imports:** Search the entire codebase (especially `src/demo/`,
   `src/harness/`, and `src/metrics/`) for any imports referencing these moved
   files and update them to point to their new location in `src.icebox...` so
   the existing experiments and demos continue to run without crashing.
4. **Add Deprecation Notices:** At the top of each moved file, add a prominent
   docstring:

```python
   """
   [ICEBOXED] - Architectural Pivot

   These modules represent an attempt to force classical, deterministic architectures to handle continuous-time biological realities (e.g., Latent Stasis, Triton kernel optimizations, and deterministic LRP). Moving forward, Stable Basin relies on natively probabilistic, energy-based thermodynamic frameworks where missing data is naturally imputed and physics-based hardware minimization renders these hacks obsolete.
   """
```

Update Documentation:

- In OPEN_CHALLENGES.md, prepend [ICEBOXED] to the headers for "The CUDA/Triton
  Quest" and the mechanistic interpretability quest regarding native H-SSM
  dictionary learning.

- In ISSUES.md, prepend [ICEBOXED] to the "MASR Mamba on Toxic Shock test"
  issue.

- Add a short note under each indicating we have pivoted away from deterministic
  CUDA/Triton development.
