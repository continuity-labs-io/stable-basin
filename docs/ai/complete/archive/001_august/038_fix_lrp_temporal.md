Context: The `MambaLRPEpsilon` class in `src/metrics/mamba_lrp.py` is failing to
propagate relevance backward through time. The resulting attribution heatmap is
just a single vertical line at the target frame.

Task: Fix the temporal unrolling logic in `MambaLRPEpsilon.attribute()`.

1. Currently, the temporal loop operates from `T-1` down to `0`. However, the
   mathematical approximation
   `a_mem = hidden_states[:, t-1, :] * retention_factor` is failing to properly
   carry the accumulated `R_total_t` backward into the _next_ iteration's
   `R_memory`.
2. Update the backward loop to ensure `R_memory` properly accumulates and passes
   relevance to the previous time step (`t-1`).
3. Increase the `retention_factor` slightly (e.g., `0.98`) to encourage longer
   temporal smearing across the 500ms window, making the buildup to the crash
   visually obvious in Panel 4.
