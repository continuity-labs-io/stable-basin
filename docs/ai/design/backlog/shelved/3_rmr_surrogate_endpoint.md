# 📄 Design Document: RMR2 Surrogate Endpoint & Combinatorial Simulation

## 1. Executive Summary
With the successful integration of the 309-dimensional Calico CATNAP dataset, Stable Basin has established a baseline of mammalian thermodynamic aging via smart-cage telemetry. To prepare for the LEV Foundation's Robust Mouse Rejuvenation 2 (RMR2) study, we must upgrade the pipeline to a Combinatorial Multi-Mechanistic Dashboard. This will allow us to simulate the exact 20 treatment arms of RMR2 in silico, translating thermodynamic rescue scores into predicted lifespan additions weeks after the mice are dosed.

## 2. Phase 1: In Silico Pharmacological Mapping
We will map the 8 LEV interventions to distinct mathematical variables within the ThermoFlowFactor and Thermostat SDE.

* **Group 1: Precision (Π) Modulators (Erase noise, restore structural steepness)**
  * Interventions: Partial Cellular Reprogramming, MSCs
  * Physics Override: Multiply the diagonal precision prior v_diag by a gain factor λ_Π. Forces the organism to route energy toward structural repair.
* **Group 2: Friction (Γ) Modulators (Membrane integrity, leak prevention)**
  * Interventions: Deuterated Fatty Acids (D-PUFAs), LC-FACS Senolysis
  * Physics Override: Increase the eigenvalues of the dissipative matrix Γ by λ_Γ, damping the system's response to random chaotic perturbations.
* **Group 3: Temperature (T) Suppressors (Reduce inflammaging / ROS)**
  * Interventions: IL-11 Inhibition, CASIN (Cdc42)
  * Physics Override: Directly reduce the stochastic noise scalar T in the Euler-Maruyama diffusion term (√(2 T_eff dt) L dW).
* **Group 4: Solenoidal (Q) Synchronizers (Systemic flow / rhythm)**
  * Interventions: Oxytocin, rMSA
  * Physics Override: Increase the gain on the skew-symmetric flow matrix Q by λ_Q, forcing biological rhythms back into high-amplitude limit cycles.

## 3. Phase 2: The 20-Arm Combinatorial Sweep
We will introduce `configs/rmr2_simulation.yaml` and a new script `12_rmr2_leaderboard.py`.

* **Define the Arms:** We will define the 20 exact treatment arms LEV is testing as boolean arrays.
* **The Base Floor:** The engine will "burn-in" on the Catnap dataset, then apply a baseline matrix shift to represent the Rapamycin + Exercise control group.
* **Simulation:** We will run all 20 combinatorial arms through the PredictiveCodingGraph. The output will be a predicted ranking (Leaderboard) predicting exactly which combination of drugs will yield the highest longevity rescue.

## 4. Phase 3: The Actuarial Bridge (CASPAR Integration)
We must convert Thermodynamic Rescue (R) into chronological months.

* **The Regression:** We will use the Catnap metadata (chronological_age and ultimate_lifespan). We will train a lightweight regression model mapping the un-intervened Hessian Trace and Langevin Variance of each mouse directly to its remaining lifespan.
* **The Output:** When we run the 20-arm sweep, the pipeline will output predictions formatted as:
  * Arm 7 (7-drug combo without IL-11): +14.2 Months Mean Lifespan Extension.
  * Arm 14 (Monotherapy MSCs): +4.1 Months Mean Lifespan Extension.
