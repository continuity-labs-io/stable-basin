**Context Files to Load / Modify:**
* `src/echo/primitives/ebm.py`

**Task: Paper 1 Physics - Enforce Waddington Structural Prior** The
`PrecisionWeightedEBM` is currently an unconstrained MLP. Because the MSE loss
only constrains the first derivative, it is learning a slightly negative
curvature (a saddle/hill) rather than a deep Waddington basin, which causes the
Hessian Trace to hover near zero and invert the aging logic. We must physically
constrain the energy landscape to be globally convex (a basin) by injecting a
biological structural prior.

**Core Objectives:**

**1. Add the Structural Prior (`ebm.py`):**
* Locate the `__call__` method of `PrecisionWeightedEBM`.
* Currently, it returns the raw MLP energy. Modify it to include a global
  quadratic prior.
* Compute the base quadratic energy: `e_prior = 0.5 * 0.1 * jnp.sum(x ** 2)`
  (using a mild coefficient like `0.1` so it doesn't overpower the MLP).
* Compute the network energy: `e_mlp` (from the MLP forward pass).
* Return the sum for the energy: `e_prior + e_mlp`.
* *Why:* This anchors the energy scale and guarantees the landscape is globally
  a positive-definite basin, allowing the MLP to focus strictly on learning
  non-linear topographical deformations (the sharp trench of the limit cycle).

**Constraints:**
* Keep the changes strictly isolated to the EBM's forward pass. Do not modify
  the rest of the file or other benchmarks.
