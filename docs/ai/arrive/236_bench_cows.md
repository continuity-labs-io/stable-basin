Now write the execution script `src/benchmarks/12_synthetic_cows_sniper.py`.

1. Import `SyntheticCOWSDataset` and create a `train_loader` (`adversarial_mode=False`) and a `test_loader` (`adversarial_mode=True` to mimic the adversarial COWS test set).
2. Initialize `PredictiveCodingGraph` from `src.echo.architecture.predictive_coding_graph`. 
3. Use our `build_graph` logic (or initialize manually) to create a `micro_observer` with `d_sensory=10` and a `macro_observer` with `d_sensory=20`. 
4. **Crucial:** Ensure both observers use `PrecisionWeightedEBM` (by setting `ebm_type='dense'`). This ensures the `HierarchicalThermoFlowFactor` calculates the precision-weighted Mahalanobis distance between the macro and micro states.
5. Write a custom JAX `scan` or unroll loop. At each step `t`, inject `x_micro[:, t]` into the micro sensory partition and `x_macro[:, t]` into the macro sensory partition, then step the graph.
6. Attach a simple classification readout (e.g., `eqx.nn.Linear`) to the final integrated `macro` state to output logits for the 35 classes.
7. Train the model using Optax/Adam. The total loss must be: `CrossEntropyLoss(logits, labels) + 0.1 * graph.ebm(concat_state)[0]`. This minimizes classification error while preserving the Joint Free Energy physics that absorbs the noise.
8. Train for a few epochs, then run the evaluation loop on the noisy/rotated `test_loader`.
9. At the end, print the accuracy to the console formatted exactly like this:
`[STABLE BASIN] randrot_noise_infer_cows_compositional (Synthetic Proxy): {accuracy:.2f}% (TBP Baseline: ~2.80%)`

*Note: If the code requires novel software library dependencies, include a dedicated 'Dependencies' section at the end of the response. Dial down the intensity of all logging, print statements, and variable names. Keep the internal tone calm, precise, and practical.*
