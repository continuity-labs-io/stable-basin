## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX and
Equinox.

## TASK

We are building `src/echo/clinic/interventions.py`. Implement the mathematical
infrastructure required to execute the Active Inference clinical workflow:
Annealing Counterfactual Twins and calculating Hierarchical Discordance.

MATHEMATICAL CONSTRAINTS & REQUIREMENTS:

- Do NOT use PyTorch. The code must be pure JAX and Equinox.
- Assume we have access to the components of our `PredictiveCodingGraph` and
  `MarkovBlanketObserver`.

1. Implement `DigitalTwinAnnealer(equinox.Module)`:
   - This class mathematically restores a degraded Digital Twin to its optimal
     physical state without needing a population database.
   - `__init__` takes no parameters (it's a stateless utility).
   - Implement a method
     `anneal_twin(self, degraded_graph: equinox.Module, gamma_boost: float = 1.5, pi_boost: float = 2.0) -> equinox.Module`.
   - Because Equinox modules are immutable PyTrees, you MUST use
     `equinox.tree_at` to modify specific parameters of the graph to create Twin
     B. Do NOT mutate the original graph in place:
     - Restore Friction (Γ): Locate the unconstrained weights of the
       `DissipativeFriction` modules (`self.W`) at both micro and macro levels.
       Multiply them by `gamma_boost` to simulate restoring the tissue's
       physical damping capacity.
     - Restore Precision (Π): Locate the weights of the `precision_head` inside
       the `PrecisionWeightedEBM` for both levels. Multiply them by `pi_boost`
       to mathematically steepen the prior and restore the tissue's
       observability.
   - Return the newly annealed Graph (Twin B).

2. Implement `DigitalTwinInterrogator(equinox.Module)`:
   - This class measures "Silent Drift" (Blindness) by comparing micro-level
     surprisal to macro-level surprisal after an active ping.
   - `__init__` takes no parameters.
   - Implement a method
     `ping_and_measure(self, graph, x_micro, x_macro, q_ext_pulse)`.
   - `graph` is a `PredictiveCodingGraph`. `q_ext_pulse` is a small exogenous
     perturbation vector (e.g. an electrical ping from the hardware).
   - Add `q_ext_pulse` to `x_micro` to simulate the bioelectric shock:
     `x_micro_shocked = x_micro + q_ext_pulse`.
   - Extract the Joint Free Energy closure (the `joint_energy_fn`) from the
     graph's logic.
   - Use `jax.grad(joint_energy_fn, argnums=(0, 1))` to extract the gradient
     with respect to the _micro_ state (micro surprisal) and the _macro_ state
     (macro surprisal) evaluated at `(x_micro_shocked, x_macro)`.
   - Compute the L2 norm of both gradients.
   - The "Discordance Score" is the ratio: `macro_norm / (micro_norm + 1e-8)`. A
     healthy system passes the prediction error up (ratio > 0.1). A system with
     Silent Drift absorbs it blindly at the micro level (ratio near 0.0).
   - Return a dictionary:
     `{"micro_surprisal": micro_norm, "macro_surprisal": macro_norm, "discordance": discordance_score}`.

## TESTING

Create `tests/echo/clinic/test_interventions.py`. Write a `pytest` suite that:

1. Mocks a degraded `PredictiveCodingGraph`.
2. Tests `DigitalTwinAnnealer`: Asserts that the returned Twin B has larger
   weight norms for friction and precision than Twin A, and that Twin A remains
   unmutated (verifying PyTree immutability).
3. Tests `BioBladeInterrogator`: Injects a ping and verifies the discordance
   score is returned as a valid float. Mocks a "blind" macro gradient (all
   zeros) and asserts the discordance score drops to near zero.

Write production-grade code with clean docstrings and unit tests. Focus strictly
on these two components.
