## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX and
computational biology.

## TASK

We are building `src/echo/benchmarks/clinical_workflow_demo.py`. This is a
standalone execution script that demonstrates the 5-Step Active Inference
clinical workflow on a simulated aging patient ("Alice").

## TECHNICAL REQUIREMENTS

- The script must be executable from the command line
  (`if __name__ == "__main__":`).
- Import the components from `src.echo.clinic.interventions`,
  `src.echo.architecture`, and `src.echo.metrics.thermal_interpretability`.

## EXECUTE THE 5-STEP WORKFLOW

1. **Listen (Initialize Degraded Alice):**
   - Instantiate a `PredictiveCodingGraph` representing Alice's currently degraded
     digital twin (Twin A). Initialize it with artificially low weights for
     Friction and Precision to simulate aging.
   - Define a resting state `x_micro` and `x_macro` using random noise.

2. **Measure Geometry (The Hessian):**
   - Instantiate the `HessianCurvatureTracker` and run it on Alice's Macro
     Observer using his resting state.
   - Log and print the mean Hessian Trace. (e.g., "STEP 2: Measuring Waddington
     Geometry... Trace = X. Attractor basin is flattened.").

3. **Ping (Detecting Silent Drift):**
   - Instantiate the `DigitalTwinInterrogator`.
   - Define a small `q_ext_pulse` vector.
   - Call `ping_and_measure` on the graph.
   - Print the Discordance Score. Implement logic:
     `if discordance < 0.1: print("DIAGNOSIS: Silent Drift Detected! Macro-level is blind to physical micro-damage.") else: print("DIAGNOSIS: Concordant.")`.

4. **Compute Counterfactual (The Reference Twin):**
   - Instantiate the `DigitalTwinAnnealer`.
   - Call `anneal_twin()` on Alice's degraded graph to generate `Twin B` (Optimal
     Alice).
   - Print confirmation that Friction (Γ) and Precision (Π) have been restored
     _in silico_ without needing a population database.

5. **Actuate (The Hardware Delta):**
   - We must calculate the restorative frequency required to heal Alice.
   - Calculate the deterministic drift (the physical force vector
     `-(Q - Γ) @ grad_E`) for the un-annealed Twin A at the current state.
   - Calculate the deterministic drift for the annealed Twin B at the current
     state.
   - The required restorative actuation vector is exactly the difference:
     `Q_actuation = drift_B - drift_A`.
   - Print the L2 norm of `Q_actuation`. Log: "This is the exact exogenous
     energy the hardware must inject to force the physical tissue back
     into its youthful limit cycle."

## OUTPUT

- The script should print a highly readable, terminal readout clearly
  separating the 5 steps.
- Use the `logging` library to create visually distinct section headers (e.g.,
  `=== Step 3: The Hardware Ping ===`).
- Generate a simple Matplotlib dashboard `output/echo/active_inference_clinical_workflow.png`
  showing 2 subplots: a) A bar chart comparing Micro vs Macro Surprisal from the
  Ping phase. b) A bar chart comparing the Hessian Trace (Steepness) of Degraded
  Alice vs. Ideal Alice.

Write production-grade, clean Python. Do not hallucinate external database
dependencies; rely purely on the physical matrices of the model.
