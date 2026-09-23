Reviewer 2 caught a fatal mathematical error in our intervention scripts: we are artificially multiplying the final evaluation metric by lambda *after* the simulation completes. We also have a bias where we impute `NaN`s as 1.0 (perfectly healthy) when the physics engine blows up.

Please fix `07_worm_gait_intervention.py`, `08_worm_gait_lambda_sweep.py`, and `09_pharmacological_translation.py`:

1. **Remove Post-Hoc Multiplication (Scripts 07 & 08):** 
   Locate where the traces are extracted and multiplied (e.g., `trace_A_batch = get_traces(...) * lambda_A` or `* lam`). Remove the `* lambda_A`, `* lambda_B`, and `* lam` entirely. The Hessian trace must evaluate the raw curvature of the resulting trajectory without artificial scaling.

2. **Fix NaN Imputation Bias (Scripts 07 & 08):**
   If the SDE blows up into `NaN`s, it means the system lost all thermodynamic stability. Change all instances of `np.nan_to_num(..., nan=1.0)` to `np.nan_to_num(..., nan=0.0)`. Update the corresponding logger warning to say "Imputing with 0.0".

3. **Re-calibrate the 4PL Fit (Script 09):**
   Because we removed the `* lambda` multiplier, the dose-response curve will no longer be a straight line through the origin. 
   - Update the `bounds` in `curve_fit` to strictly constrain `ec50` between `min(lambdas)` and `max(lambdas)` (instead of 0.1x to 10.0x). 
   - Remove the `* lambda_base` and `* lambda_rescue` multiplication from the `G_baseline` and `G_rescue` calculations.
