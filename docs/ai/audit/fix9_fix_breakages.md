Fix breakages from the naming cleanup. No other changes.

1. killifish_task.py and catnap_task.py do not parse: render_animation has no body and
   cohort_labels contains unreachable code. Give render_animation a body that logs "animation
   not implemented" and returns. Make cohort_labels a separate property returning only the tuple.
2. 05_aging_ebm.py: plot_ablation_results uses label_a and label_b, which exist only in main.
   Pass them in as arguments.
3. Makefile: aging-resilience-fit-lambda, -lambda-comparison and -lambda-argmax call
   06_infer_fitted_lambda, 07_intervention and 09_pharmacological_translation, which no longer
   exist. Point them to 06_fit_lambda, 07_lambda_comparison and 09_lambda_argmax.
4. Remove silent fallbacks to synthetic data in WormGaitTask.get_raw_datasets and in load_ts
   (11_null_control.py). Raise FileNotFoundError instead. Allow the mock only behind an explicit
   --mock flag, and write "mock" into every output file name and JSON when it is used.
5. diagnostic_engine.py: remove the hardcoded confidence_score (0.98 / 0.50) and the "mechanism"
   field guessed from name prefixes.
6. threshold_monitor.py: rewrite the docstrings and log messages to describe threshold checks and
   logging only. Remove IV pump, infusion, therapy, flight computer and homeostasis.

Gates:
- python -m py_compile succeeds on every .py file under src/.
- make -n succeeds for every Makefile target.
- pytest pass count is not lower than before the change.
