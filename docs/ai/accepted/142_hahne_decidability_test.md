ROLE: You are an elite Scientific Machine Learning Engineer.

TASK: Create `src/echo/benchmarks/bioblade_diagnostic.py`. This script
demonstrates [UPDATE 003]. It tests whether an aging biological system has
suffered an irreversible structural collapse (Reachability Failure) or merely a
sensory/informational blindness (Policy Failure) by injecting an active
Bio-Blade ping.

TECHNICAL REQUIREMENTS:
- The script must be executable from the command line (`if __name__ ==
  "__main__":`).
- Setup:
  - We need two degraded Observers (e.g., `d_internal=4, d_sensory=4,
    d_active=4, d_external=4`), initialized at `x_init=zeros`.
  - **Patient A (Blindness):** Has a healthy Engine & Brakes, but zeroed-out
    sensors. (`D_s = jnp.zeros((4, 4))`).
  - **Patient B (Structural Collapse):** Has healthy sensors (`D_s = None`), but
    its physical Brakes are destroyed. Use `equinox.tree_at` to multiply
    `observer.dissipative.W` by `0.001` (destroying friction/damping) and
    `observer.solenoidal.W` by `0.001` (destroying rotation).

- The Simulation (`seq_len=200`, `dt=0.01`):
  - **Phase 1 (Endogenous Failure):** Apply a harsh environmental drift
    `omega_seq` to both patients using `forced_unroll` (e.g., a drift of `2.0`
    on all dimensions). Because Patient A is blind and Patient B is structurally
    weak, BOTH will drift far away from the homeostatic origin.
  - **Phase 2 (The Bio-Blade Intervention):** We design a therapeutic `q_seq`
    that acts as a simple exogenous proportional controller pushing the state
    back to zero. For the drifted state `x_drifted`, calculate `q_pulse = -5.0 *
    x_drifted`, broadcast this into a `q_seq`, and unroll BOTH observers again
    starting from `x_drifted` to fight the `omega_seq`.

- The Logic Gate (The Core Output):
  - Measure the final distance to the origin (`L2 norm`) for both patients after
    the Phase 2 intervention.
  - Set a `rescue_threshold = 2.0`.
  - For each patient, implement the exact logic gate: `if distance <
    rescue_threshold:` `print("[DIAGNOSIS: POLICY_OBSERVABILITY_FAILURE]
        Hardware restored homeostasis. Physical Waddington basin is intact.")`
    `else:` `print("[DIAGNOSIS: REACHABILITY_COLLAPSE] Hardware failed. The
        physical attractor basin is irreversibly destroyed.")`
  - Patient A should pass the rescue (Policy Failure). Patient B should fail the
    rescue (Reachability Collapse).

- Output:
  - Generate a Matplotlib figure `output/echo/bioblade_diagnostic.png`.
  - Create 2 subplots (one for Patient A, one for Patient B).
  - On each subplot, plot the Physical State Divergence (L2 Norm) over time,
    connecting Phase 1 and Phase 2.
  - Add a vertical dashed line labeled "Bio-Blade Actuation Triggered" where
    Phase 2 begins.
  - Add titles clearly indicating the final Diagnosis.
  - Cite Hahne (2026) "Beyond Return to Baseline: Reachability, Observability,
    and the Measurement of Physiological Margin". 

TESTING: Create `tests/echo/benchmarks/test_bioblade_diagnostic.py` that imports
`main()`, mocks `matplotlib.pyplot.savefig`, and asserts execution without JAX
errors.
