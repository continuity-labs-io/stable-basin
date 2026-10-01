# Adversarial Code Coverage Audit: Time Domain Metrics

## Objective
We have identified critical sections of the codebase that lack test coverage. Your primary task is to write adversarial unit tests designed specifically to **break** these uncovered code paths and expose latent bugs. Do not write happy-path tests simply to increase the coverage percentage; your goal is to find vulnerabilities.

## Instructions
1. **Adopt the Paranoid Debugger Persona**: Act as an adversarial tester in `[MODE: PARANOID_DEBUGGER]`. Actively look for ways the code could fail under pressure.
2. **Target the Weak Points**: Focus on edge cases and failure modes, such as:
   - Extreme inputs (e.g., highly sparse data, extremely large/small values, out-of-bound variables).
   - Shape, dimensionality mismatches, or unintended broadcasting.
   - Singularity conditions (e.g., division by zero, NaN propagation, Infs).
   - Invalid or unexpected state transitions.
3. **Analyze and Fix**:
   - If your tests expose a bug, clearly explain what the bug is, the root cause, and how your adversarial test exposed it.
   - Propose and implement a robust fix for the bug in the source code.
4. **Adhere to Testing Rules**:
   - **Unit Test Structure**: Strictly delineate all tests with `ARRANGE`, `ACT`, and `ASSERT` blocks.
   - **1-to-1 Invariant Rule**: For every non-standard tensor operation, there must be an isolated mathematical invariant test.
   - **Paranoid Debugging**: Use `torch.autograd.set_detect_anomaly(True)` at the top of the test script and ensure tests run on `device="cpu"` when investigating NaNs.

## Critical Uncovered Code
Please investigate and write adversarial tests for the following file and specific lines:
- [ ] File: `src/metrics/time_domain.py` (Missing: 58, 85, 171, 181-182, 212-226, 248, 273, 283-284, 297-299, 354-364, 379-406, 420-430)

**Targeted Attack Vector:** This file has massive gaps related to sliding windows and correlations. Attack the "Lag Paradox": pass a `tau` (lag) value that is larger than the entire time series sequence length. Test sliding windows where `window_size` is not cleanly divisible by the sequence length. Force a `ZeroDivisionError` on correlations by passing constant (zero-variance) data.
