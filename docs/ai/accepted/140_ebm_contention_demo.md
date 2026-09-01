ROLE: You are an elite Scientific Machine Learning Engineer.

TASK: Create `src/echo/benchmarks/concurrent_contention_benchmark.py`. This script demonstrates [UPDATE 002]: how a biological system fails non-linearly due to "Contention" when multiple simultaneous demands overwhelm shared pathways.

TECHNICAL REQUIREMENTS:
- The script must be executable from the command line (`if __name__ == "__main__":`).
- Setup:
  - Instantiate a single `MarkovBlanketObserver` (`d_internal=8, d_sensory=8, d_active=8, d_external=8`, total `d_state=32`).
  - To simulate a living tissue with a restorative basin without running a massive training loop, mathematically boost its friction to ensure it actively dampens perturbations: 
    `observer = eqx.tree_at(lambda o: o.dissipative.W, observer, observer.dissipative.W * 5.0)`

- The Simulation (3 Phases, `seq_len=200`, `dt=0.01`):
  - **Phase A (Stressor 1):** Define `omega_1`, an external force vector (shape `200, 32`) that applies a constant force of `10.0` ONLY to the `sensory` dimensions. Unroll the observer from the origin (`x_init=zeros`) using `forced_unroll(..., seq=None, omega_seq=omega_1)`.
  - **Phase B (Stressor 2):** Define `omega_2`, applying a constant force of `10.0` ONLY to the `active` dimensions. Unroll from origin using `omega_seq=omega_2`.
  - **Phase C (Contention):** Define `omega_3 = omega_1 + omega_2`. Unroll from origin using this combined force `omega_seq=omega_3`.

- Metrics & Output:
  - Extract the trajectory for all 3 phases.
  - Compute the Physical State Divergence (L2 Norm of `x_current - x_init`) over time for A, B, and C.
  - Instantiate `HessianCurvatureTracker` and calculate the Trace of the Hessian (the Steepness/Resilience) for the final state of each phase.
  - Generate a Matplotlib figure `output/echo/contention_benchmark.png`.
    - Plot the Divergence for Phase A, Phase B, and Phase C on the same graph.
    - **The Proof:** Because the MLP backbone is non-linear and pathways are shared, Phase A and B should reach a relatively stable, finite plateau (the amplified friction successfully counteracts the single force). Phase C should explode non-linearly (the divergence goes way beyond the simple sum of A and B) because the system's shared pathways are overwhelmed (Contention).
  - Print a terminal readout comparing the Final Divergence and Final Hessian Trace for A, B, and C.

TESTING:
Create `tests/echo/benchmarks/test_concurrent_contention_benchmark.py` that imports `main()`, mocks `matplotlib.pyplot.savefig` to prevent file IO, and asserts execution without JAX errors.
