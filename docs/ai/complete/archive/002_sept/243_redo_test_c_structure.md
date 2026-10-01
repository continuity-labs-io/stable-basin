Context: src/benchmarks/aging_resilience/12_structure_checks.py. Test C returned closure = 6.0
(test MSE_full 0.331 vs MSE_macro 0.055). The full past contains the macro past, so a correctly
fitted full model should score near or below the macro-only model on held-out worms. Fix and
rerun Test C only. Do not change Tests A or B.

1. Build lagged features within each window only; lags must not cross window boundaries.
2. Standardize features using training-worm statistics only. Fit RidgeCV with alphas
   logspace(-3, 3, 13), using grouped CV by worm on the training worms.
3. Report train and test MSE for both models. Sanity check: train MSE_full <= train MSE_macro,
   because the full model contains the macro model. If this fails, stop and report: the fitting
   code is wrong.
4. Report closure = test MSE_full / test MSE_macro with a bootstrap 95% CI over test worms.
5. Decoupling check: rerun the forced unroll for each test worm with its sensory input replaced
   by another worm's input (same windows, same PRNG keys). For macro internal dims, report
   mean |change| divided by the SD of the original trajectory.
6. Record in the output which data files were loaded and their sha256.

Gate (fixed before running):
- PASS: closure point estimate >= 0.9, CI upper bound <= 1.1, and decoupling change >= 0.1.
- FAIL: closure < 0.9 (micro detail adds information; macro is not closed).
- VOID: closure > 1.1 after these fixes (comparison still broken), or decoupling change < 0.1
  (macro ignores the data, so closure is trivial).
Write structure_checks_c_v2.json. Report effect sizes and CIs, no p-values.
