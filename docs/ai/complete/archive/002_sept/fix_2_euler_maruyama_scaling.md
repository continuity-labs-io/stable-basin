# Fix 2: Correct Euler-Maruyama Noise Scaling in Reaction-Diffusion Vessels

**Severity:** High

## Description
Brownian noise is incorrectly scaled by dt squared instead of the physically required square root of dt, causing thermodynamic noise to vanish incorrectly and breaking the Fluctuation-Dissipation Theorem.

## AI Execution Plan
Develop a plan to correct the stochastic integration by adding noise directly during the state update step, scaled by the square root of dt. Get it reviewed, implement, ensure tests pass, and commit.
