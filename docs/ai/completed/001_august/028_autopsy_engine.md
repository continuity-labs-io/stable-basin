# 028: Diagnostic Engine

## Objective

Create a new Python module named `src/metrics/diagnostic_engine.py` that translates
raw continuous neural tracking data and MambaLRP attribution tensors into the
structured Thermodynamic Diagnostic JSON payload.

## Context

Our continuous monitoring system tracks a 114-Dimensional biological state
vector per time step (100D Sigma morphological shape indices, 12D Psi RNA
expression anchors, and 2D Omega bioelectric voltage traces). When a Waddington
crash or critical anomaly is detected via thermodynamic metrics (like KSM), this
engine executes a backward attribution trace and generates an automated
diagnostic diagnostic report.

## Requirements

1. **Class `ThermodynamicDiagnosticEngine`**:

   - `__init__(self, model, feature_names=None)`: Stores the trained model and
     an optional list of 114 feature names. If `feature_names` is None,
     auto-generate them matching the Trifecta layout: `Sigma_PC001` to
     `Sigma_PC100`, followed by 12 RNA anchors (`Psi_NFE2L2`, `Psi_TP53`,
     `Psi_CDKN2A`, `Psi_TREM2`, `Psi_APOE`, `Psi_IL6`, `Psi_GFAP`, `Psi_MAPT`,
     `Psi_NANOG`, `Psi_CASP3`, `Psi_CAS13`, `Psi_GAPDH`), and 2 voltage tracks
     (`Omega_VoltRed`, `Omega_VoltGrn`).

2. **Method
   `generate_diagnostic(self, x_sequence, crash_time_step, confidence_score=0.98)`**:

   - Accept a batched input tensor `x_sequence` of shape `[1, Time, 114]` and
     the designated `crash_time_step`.
   - Compute the first-order Taylor attribution (Input \* Gradient) across the
     sequence leading up to the crash using the model's forward pass.
   - Identify the primary latent driver dimension with the highest global
     variance or gradient magnitude.
   - Extract the top 3 critical time steps prior to the crash (e.g.,
     $T-2, T-4, T-7$ or specific inflection points). For each time step,
     identify the highest-scoring `flagged_input` feature and its normalized
     `relevance_score`.
   - Map each flagged feature to a human-reaksmle biological `mechanism` string
     based on its modality (e.g., if a `Psi` gene triggers, describe it as an
     RNA stress alarm; if `Omega` flickers, describe it as an electrical
     baseline destabilization).

3. **Output Schema**:

   - Return a standard Python dictionary strictly matching the user's Diagnostic
     JSON schema:
     ```json
     {
       "status": "CRITICAL_FAILURE_PREDICTED",
       "predicted_crash_time": "T={crash_time_step}",
       "confidence_score": 0.98,
       "anomaly_ontology": {
         "primary_latent_driver": "...",
         "causal_trace": [
           {
             "time_step": "T=...",
             "flagged_input": "...",
             "relevance_score": 0.0,
             "mechanism": "..."
           }
         ]
       }
     }
     ```

4. **Execution Test Block**:
   - Include a `if __name__ == "__main__":` block that initializes a mock
     `SpikeForecaster` or `NeocorticalEngine` (or similar dummy model), passes a
     dummy 114-D sequence with an injected crash, runs `generate_diagnostic()`, and
     prints the pretty-printed JSON string to the terminal to verify zero-error
     execution.
