Generate an update to `README.md` for the newly refactored `stable_basin` open-source PyTorch library. 

Constraints:
1. Title: Stable Basin
2. Subtitle: Continuous-time state space models (SSMs) and latent thermodynamics for irregular time-series.
3. Pitch it strictly as a foundational primitive for ML engineers handling messy, asynchronous data (dropped sensors, NaNs) and predicting catastrophic model drift before it happens.
4. Include minimal, beautiful Python code snippets showing how dead-simple it is to use:
   - `MaskAwareMamba` (for dropped sensors)
   - `EarlyWarningRadar` (for predicting model crashes)
   - `MambaLRPEpsilon` (for exact Mamba LRP interpretability)
5. Include an "Origins / Mission" section at the very bottom. This is the Trojan Horse. Explain that this math was originally forged to solve human aging and map biological phase transitions. Provide a link to the `/examples/aging_research` folder, inviting engineers who want to solve biology to check out the demos and open problems.
