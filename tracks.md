| Track | Question it answers | Link to the North Star | Status |
|---|---|---|---|
| 1. Lifespan benchmark | Do time-series features predict remaining lifespan beyond static summaries, in held-out animals? | Is the North Star | Active, lead |
| 2. Model-free resilience metrics | Recovery time, lag-1 autocorrelation, variance, entropy production from raw data | Candidate predictors for track 1 | Active |
| 3. Echo model (energy-based SDE) | Does a fitted dynamical model give better predictors or interpretable resilience measures? | Competes in track 1; must beat track 2's features | Active, gated on track 1 |
| 4. Validation code | Null controls, positive controls, cohort splits, leakage checks | Keeps every track honest | Active, always |
| 5. MEA tissue QC | Spike sorting, drift, longitudinal comparability on MaxWell | Commercial Vector 1; reuses tracks 2 and 4 | Active, separate business decision |
| 6. Sequence models, ML learning | SSMs, Mamba, transformers, sensor fusion | Possible encoder for track 3 at high dimension; otherwise learning | Parked; learning time only |
| 7. Thermodynamic hardware | Torx/Extropic compatibility | None until the 2027 hardware and a sampling-heavy workload | Parked |
| 8. Control and interventions | Controllers, the Gymnasium | Needs a validated surrogate first | Parked |
