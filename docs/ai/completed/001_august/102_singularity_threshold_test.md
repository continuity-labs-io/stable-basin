# Singularity Threshold Test

### Prompt 2: The Singularity Threshold (The Dissonance Check)

**Target File:** `tests/models/ssm/test_masr_mamba_rigor.py`
**Context Files:** `src/models/ssm/masr_mamba.py`

**Instructions:**
Create a rigorous boundary test named `test_mamba_masr_singularity_prevention`.
1. The denominator for `B_bar` in `mamba_masr_reference_scan` contains a potential division-by-zero vulnerability: `(A - 1e-8)`. 
2. Deliberately force the continuous parameter `A` to be adversarial values: exactly `1e-8`, `0.0`, and `-1e-9`. Do this in separate sub-tests or loop iterations.
3. Run the reference scan for each condition with standard valid floats for all other parameters.
4. Assert strictly that the output tensor contains absolutely NO `NaN` or `Inf` values (`torch.isnan(y).any()` and `torch.isinf(y).any()` must both be False). 
The test must prove the epsilon stabilizer prevents a silent numerical explosion during edge-case integration.
