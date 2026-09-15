# Fix 1: Fix Markov Blanket Topology Violation in Dissipative Friction

**Severity:** Critical

## Description
Eigenvalue clipping destroys the sparsity pattern of the mask, causing internal and external states to become directly coupled and violating the Markov Blanket conditional independence theorem.

## AI Execution Plan
Develop a plan to fix the topology violation by removing eigenvalue clipping and applying the topology mask directly to the lower-triangular Cholesky factor. Get it reviewed, implement, ensure tests pass, and commit.
