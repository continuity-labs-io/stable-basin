Reviewer 2 pointed out that scoring our therapeutic rescue by multiplying the Hessian trace by lambda is mathematically circular. Because lambda acts as an inverse-temperature scaling, we must score the rescue based on distributional distance in the observable space.

Please overhaul `07_worm_gait_intervention.py`, `08_worm_gait_lambda_sweep.py`, and `09_pharmacological_translation.py`:

1. **Remove Circular Math:** Delete all instances of multiplying traces by lambda (e.g., `* lambda_A`, `* lambda_B`, `* lam`). Also change any `nan_to_num(..., nan=1.0)` to `nan=0.0`.
2. **Implement Energy Distance Metric R(λ):** Import `energy_distance` from `scipy.stats`. 
   - Let `Y` be the clean, held-out biological data (flatten dimensions 0-5 of a sequence from `eval_young_loader`).
   - Let `M_lam` be the flattened observable dimensions (0-5) of the simulated trajectory at a given lambda.
   - Calculate $R(\lambda) = 1 - \frac{D(Y, M_\lambda)}{D(Y, M_{\text{baseline}})}$, where $D$ is the `energy_distance` and $M_{\text{baseline}}$ is the simulation at the pathological lambda.
3. **Update Plots & Fitter:** 
   - In Script 07, update Panel C to show $R(\lambda)$ instead of the Hessian trace.
   - In Script 08, plot $R(\lambda)$ vs $\lambda$. Rename axes from "Precision Injection Parameter" to "Inverse-Temperature Scaling ($\lambda$)".
   - In Script 09, fit the 4PL Hill Equation to the $R(\lambda)$ curve. Constrain the `ec50` bounds to strictly fall between `min(lambdas)` and `max(lambdas)` and remove the `Delta G` thermodynamic translation calculations.
   
