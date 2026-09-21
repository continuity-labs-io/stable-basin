# Adversarial Code Coverage Audit: Entropy Production

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
- [ ] File: `src/metrics/entropy_production.py` (Missing: 64, 82-110, 134, 146-150, 164-165, 171-173, 212, 244-248, 251-254, 276-285)

**Targeted Attack Vector:** Attack the covariance matrices and MOU transition estimators. Force singular (non-invertible) matrices, pass zero-variance signals, or pass perfectly symmetric trajectories to try and break the Cholesky decompositions and linear EP calculations.
