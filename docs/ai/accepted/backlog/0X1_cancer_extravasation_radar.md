Create a new sim2real dataloader and visual simulation script for my biological physics engine. We are mocking 'Cancer Extravasation' based on LLSM imaging.

1 In src/data/sim2real/sigma_phase_structure_dataloader.py, create ExtravasationSigmaLoader. Generate a 3000-step 100-D latent sequence ($\Sigma$).

Phase 1: Rolling (T=0 to 1000): Fast, stable oscillation with low variance.

Phase 2: Adhesion (T=1000 to 2000): The cell sticks. Inject 'Critical Slowing Down'—variance slowly rises, and Lag-1 Autocorrelation (AR1) approaches 1.0.

Phase 3: The Breach (T=2000 to 3000): Massive variance explosion. The L2 norm of the vector instantaneously jumps 50%.

2 Create src/demo/12_extravasation_radar_sim.py. Pass this sequence through my existing ThermodynamicMetrics class (calculate_csd and calculate_ksm).

3 Generate a 3-panel dark-mode Matplotlib dashboard. Top: 100-D Heatmap showing the 50% jump at T=2000. Middle: CSD Curve showing wobble peaking in Phase 2. Bottom: KSM Curve dropping below 0.85 in Phase 2.

4 Plot a red dashed line where the KSM triggers the 'Early Warning Radar', proving we detect the metastasis before the physical breach in Phase 3.
