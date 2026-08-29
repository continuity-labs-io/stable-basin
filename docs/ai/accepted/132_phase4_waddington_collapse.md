## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX,
Equinox, PyTorch interoperability, and computational biology.

## TASK

We are building `src/echo/benchmarks/waddington_collapse.py`. Implement a
standalone evaluation script that serves as the "Waddington Collapse" real-data
benchmark. This script will bridge our JAX-based ECHO architecture with the
existing PyTorch `PharmacologicalShockDataset`, execute a toxic shock scenario,
and calculate the Energy Basin Escape Time (EBET).

MATHEMATICAL CONSTRAINTS & THE PROOF:

1. Biological collapse is defined thermodynamically as the flattening of the
   tissue's macroscopic prior (the attractor basin).
2. The benchmark computes the Energy Basin Escape Time (EBET). EBET is the time
   difference between when the Hessian Trace drops below a critical threshold
   (e.g., 50% of its healthy baseline) and when the actual physical signal
   collapses (the ground-truth electrical crash).
3. The benchmark passes if EBET > 0. The physical curvature of the neural
   network's beliefs MUST flatten BEFORE (or exactly when) the electrical
   signals actually stop firing, proving that aging/death is a top-down control
   failure.

## TECHNICAL REQUIREMENTS

- The script must be executable from the command line
  (`if __name__ == "__main__":`).
- **Data Loading & Interoperability:**
  - Import and instantiate
    `PharmacologicalShockDataset(condition="50uM", seq_len=1000)` from
    `src.data.ephys.pharma_shock_dataset`. Assume `input_dim = 1024` for the
    HD-MEA data.
  - Extract a single sequence tensor from the dataset (e.g., shape
    `[1000, 1024]`).
  - Convert the PyTorch tensor to a JAX array safely:
    `jnp.array(tensor.numpy())` to avoid memory allocator clashes.
- **Architecture Setup:**
  - Instantiate a Micro `MarkovBlanketObserver` where `d_sensory` matches the
    HD-MEA input dimension (1024). Set internal, active, external dimensions to
    sensible defaults (e.g., 64).
  - Instantiate a Macro `MarkovBlanketObserver` operating on a smaller,
    condensed latent space (e.g., `d_state=32`).
  - Instantiate the 2-level `PredictiveCodingGraph` wrapping both observers.
  - Instantiate the `HessianCurvatureTracker` (from
    `src.echo.metrics.thermal_interpretability`) and attach it specifically to
    the Macro Observer's EBM.
- **Execution Loop:**
  - Simulate the continuous trajectory by feeding the data into the Micro
    Observer's sensory nodes. (For this benchmark script, you can use the
    initialized/untrained graph weights and pass the sequence through to
    generate the unrolled latent trajectory, or write a tiny dummy training loop
    for burn-in).
- **Metric Extraction & EBET Calculation:**
  - Compute a rolling variance of the raw HD-MEA data to find the
    `electrical_crash_frame` (when rolling variance drops below a defined
    threshold).
  - Pass the extracted Macro-State trajectory into
    `HessianCurvatureTracker.batch_calculate_curvature()`.
  - Analyze the returned `hessian_trace` array to find the
    `thermodynamic_collapse_frame` (when the trace drops below 50% of the mean
    of the first 100 frames).
  - Calculate and print:
    `EBET = electrical_crash_frame - thermodynamic_collapse_frame`.
- **Output:**
  - Generate a Matplotlib plot (saved to `output/echo/waddington_collapse.png`)
    with two subplots sharing the x-axis (Time):
    1. The raw HD-MEA telemetry variance over time (visualizing the physical
       crash).
    2. The Macro Hessian Trace over time (visualizing the flattening of the
       Waddington basin). Add vertical lines to both indicating
       `thermodynamic_collapse_frame` and `electrical_crash_frame`.

## TESTING

Create `tests/echo/benchmarks/test_waddington_collapse.py`. Write a rigorous
`pytest` suite that:

1. Synthetic Crash Test: We cannot rely on downloading massive HDF5 datasets
   during CI/CD. Create a synthetic PyTorch dataset mocking the shock (a tensor
   of shape `(200, 1024)`). Frames 0-100 are high-variance sinusoidal noise
   (healthy). Frames 100-200 are flat zeros (crash). Thus
   `electrical_crash_frame` should be calculated around 100.
2. Because the network weights are randomly initialized and not fully trained in
   the test, the Hessian trace won't naturally align with the data crash.
   Therefore, the test MUST mock or override the `HessianCurvatureTracker`'s
   return values (using `unittest.mock.patch` or similar) to simulate a
   thermodynamic collapse at frame 80 (20 frames before the data crash).
3. Call the metric calculation logic and assert that `EBET == 20`.
4. Assert that the script executes without throwing shape or JAX/PyTorch
   interoperability errors, and that the plot file is generated in a temporary
   directory.

Write production-grade code with clean docstrings. Focus strictly on this
integration script, importing the necessary modules from `src.echo` and
`src.data`.
