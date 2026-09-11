ECHO Worm Gait: The Benchmark Script

Context: Prompts 1-2 built the data layer and the harness
(EchoTrainer/EchoRunner + GaussianEBM). This prompt builds the actual
benchmark: run() twice (gaussian vs mlp), plot the comparison from
Section 5 of the design doc, print the summary table. This file should
contain ONLY orchestration and output generation — no physics, no
training, no EBM construction. If you find yourself writing any of
those, that logic belongs back in echo_runner.py, not here.

Read these first:
  - src/echo/harness/echo_runner.py   (you're extending this — see
    step 0 — before touching anything else)
  - src/echo/benchmarks/01_waddington_collapse.py (matplotlib style,
    FileNotFoundError graceful-skip pattern, output path conventions)
  - configs/echo_worm_gait.yaml       (from Prompt 2 — you'll add one
    field)

STEP 0 — Two required extensions to echo_runner.py (do these first):

  a) load_dataset(config): when data.type == "synthetic_worm_gait",
     pass config["seed"] through as the dataset's own `seed` argument
     instead of relying on SyntheticWormGaitDataset's internal default.
     Without this, two run() calls with the same top-level seed can
     still draw different synthetic worms, which would silently
     invalidate the gaussian-vs-mlp comparison.

  b) run_validation_hook(bundle, cohort_sequences, eval_key): change
     its return type from dict[str, jax.Array] to
     dict[str, dict[str, jax.Array]], where each cohort's dict is now
     {"hessian_trace": array[n_seq, seq_len],
      "macro_trajectory": array[n_seq, seq_len, d_macro]}
     with axis 0 index-aligned (macro_trajectory[i] is the trajectory
     that produced hessian_trace[i] — same forced_unroll call, don't
     regenerate it separately). Get macro_trajectory as
     trajectory[:, bundle["graph"].d_micro:] from the same
     forced_unroll call you're already making for the Hessian tracker.
     Update run(config)'s return key from "hessian_traces" to
     "validation" to reflect the richer contents, and fix any Prompt-2
     tests that reference the old key/shape.

  c) Add one field to configs/echo_worm_gait.yaml under `evaluation`:
       n_eval_sequences: 5
     and have run(config) pass the first n_eval_sequences items from
     each cohort's Subset into run_validation_hook (fewer is fine if a
     cohort has fewer items — don't error, just use what's there).

STEP 1 — src/echo/benchmarks/06_worm_gait_decline.py

  1. argparse: --config (default "configs/echo_worm_gait.yaml").
     Load the YAML once. Build two variant configs via
     copy.deepcopy(base_config), overriding ONLY
     architecture["ebm_type"] to "gaussian" and "mlp" respectively —
     ignore whatever ebm_type happens to be in the file. Both variant
     configs must share the identical top-level `seed` (don't touch
     it) so init states, eval sequences, and (per step 0a) synthetic
     data draws are identical between the two runs; the only intended
     difference is the EBM.

  2. For each variant, call echo_runner.run(variant_config). Catch
     FileNotFoundError (only possible if data.type == "celegans_gait"
     and the real database isn't downloaded) exactly like
     01_waddington_collapse.py does — print a clear message and return
     early rather than crashing.

  3. Build the figure with matplotlib's subplot_mosaic:

       fig, axes = plt.subplot_mosaic(
           [["phase_g_young", "phase_g_old"],
            ["phase_m_young", "phase_m_old"],
            ["hessian", "hessian"],
            ["loss", "loss"]],
           figsize=(10, 14),
       )

     - Rows 1-2 (phase portraits): for each of the 4 cells, plot
       macro_trajectory[:, :, 0] vs macro_trajectory[:, :, 1] for every
       sequence in that (variant, cohort)'s result, one line per
       sequence with alpha=0.5 so overlaps are visible. Titles:
       "Gaussian EBM — Young", "Gaussian EBM — Old", "Learned EBM —
       Young", "Learned EBM — Old". Axis labels "Macro dim 0" / "Macro
       dim 1".
     - "hessian" panel: 4 lines, mean hessian_trace across the
       n_eval_sequences axis plotted against time step, for
       {gaussian,mlp} x {young,old}. Use linestyle to encode cohort
       (solid=young, dashed=old) and color to encode variant, with a
       legend. This is where Section 5.2's prediction should be
       visually obvious: the two gaussian lines should sit on top of
       each other as flat lines.
     - "loss" panel: 2 lines, loss_history for gaussian and mlp vs.
       epoch, with a legend.

  4. Compute and print the summary table (Section 5.3 of the design
     doc). For each variant, for each cohort, compute the per-sequence
     mean hessian trace (mean over the time axis, giving one scalar
     per sequence), then the cohort mean and std of those per-sequence
     scalars. Build a small pandas DataFrame with columns:
       ebm_type, mean_hessian_young, mean_hessian_old, gap,
       gap_as_expected, non_trivial
     where gap = mean_hessian_young - mean_hessian_old,
     gap_as_expected = gap > 0 (our framing predicts young has higher
     curvature), and non_trivial = abs(gap) > pooled_std, where
     pooled_std is the std of the concatenated young+old per-sequence
     means for that variant. Print with df.to_string(index=False).
     State explicitly in a code comment that "non_trivial" is a rough
     heuristic appropriate for a tiny-N shakedown, not a real
     significance test — if this pipeline moves to the reprogramming
     data, that comparison should get a real statistical test instead.

  5. Save the figure to config["evaluation"]["output_plot"] (os.makedirs
     the parent dir first) and save a JSON file at the same path with
     ".json" replacing ".png" containing: the summary DataFrame (as
     records), and per-variant loss_history. Follow the NpEncoder
     pattern from src/harness/clinical_diagnostic_runner.py for
     JSON-serializing any numpy/jax scalars.

  6. if __name__ == "__main__": main()

Tests / acceptance:
  - Run the script end-to-end against configs/echo_worm_gait.yaml
    with data.type left as "synthetic_worm_gait" (don't require real
    Zenodo data for this). Confirm both the .png and .json land in
    output/echo/.
  - Add one integration-level sanity assertion (can live in a test
    file, or just be printed and eyeballed once): the gaussian
    variant's mean_hessian_young and mean_hessian_old should be equal
    to within a small tolerance — this is the same Section 5.2 identity
    as Prompt 2's unit test, now verified at the full-pipeline level
    instead of in isolation. If it's NOT close, something upstream
    (most likely step 0b's trajectory/trace alignment) is wrong — don't
    treat it as an interesting finding about gaussian EBMs.

Non-goals: no Ray, no W&B, no argparse flags beyond --config. If you
want those later, they're a separate follow-up, not part of this
benchmark's job.
