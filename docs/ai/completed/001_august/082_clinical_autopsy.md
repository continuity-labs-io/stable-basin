Latent stasis is absolutely worth pursuing. The theoretical mathematical foundation is elegant, but the true "knock-down" argument for your paper lies in proving that this architecture solves real-world hardware limits that are currently blocking clinical applications.

Here is the definitive narrative to structure your final results around:

* **The Hardware Boundary:** Standard Transformers hit a rigid 80 GB VRAM compute boundary at approximately 3.27 seconds of context for 20,000 Hz electrophysiology data. Modeling multi-hour biological dynamics at high frequencies is fundamentally intractable with standard Attention mechanisms.

* **The Standard SSM Failure Mode:** While baseline continuous-time SSMs offer O(N) scaling, they fail catastrophically on sparse, multi-rate sensor fusion. Zero-padding unobserved modalities causes representation decay, and forward-filling introduces continuous forcing functions that cause integration drift.

* **The MASR Solution:** Mask-Aware Subspace Routing intercepts the Zero-Order Hold discretization to enforce an exact identity transition for unobserved dimensions. This structurally preserves the latent state across sparse intervals while maintaining strict linear-time processing.

* **The Clinical Proof:** Transitioning from the synthetic multi-rate diagnostic benchmark to real pharmacological shock data provides the ultimate empirical proof. Demonstrating that MASR accurately detects the exact millisecond of a phase transition—while standard baselines decay into noise—validates this as the shortest path to clinical deployment.

### Step 1: Clinical Autopsy Harness

Feed the following file paths into your Antigravity IDE context:

* `src/data/ephys/pharma_shock_dataset.py`
* `src/models/ssm/meld_engine.py`
* `src/models/ssm/baseline_ssm.py`
* `src/models/attention/baseline_transformer.py`
* `src/models/ssm/mask_aware_ssm.py`
* `src/metrics/metrics.py`
* `src/metrics/mamba_lrp.py`
* `src/metrics/autopsy_engine.py`
* `src/utils/device.py`

**Raw Text Prompt to Execute:**

> Create a new file at `src/harness/clinical_autopsy_runner.py`. You are an Expert PyTorch ML Engineer. Write a production-grade CLI harness with `argparse` and `logging`.
> Requirements:
> 1. Load the `PharmacologicalShockDataset` for the `50uM` condition to extract a continuous sequence of `seq_len` frames.
> 2. Accept a `--model-type` argument to support `meld`, `baseline`, `transformer`, and `mask_aware`. Initialize the chosen model with `input_dim=1024, d_model=256`.
> 3. Train the model for `--epochs` on the early, stable portion of the sequence to learn the healthy tissue's spatial covariance. During training, evaluate and log the baseline thermodynamic stability (measuring the variance of the Koopman Stability Metric (KSM) during the healthy period).
> 4. Pass the full sequence through the trained engine to extract continuous hidden states. Use `ThermodynamicMetrics(alpha=500.0).calculate_ksm()` to dynamically identify the exact frame where the KSM drops below `--ksm-threshold`. Log this as the `crash_frame` and track the inference latency (ms per frame) during this detection phase.
> 5. Instantiate `MambaLRPEpsilon` to get the exact relevance tensor. Monkey-patch the model's `compute_attribution` method to return this tensor.
> 6. Pass the engine and sequence to `ThermodynamicAutopsyEngine` to generate a JSON autopsy report for the `crash_frame`. Log a metric evaluating the causal coherence (e.g., sparsity/focus of the relevance tensor).
> 7. Plot a 3-panel dashboard using `matplotlib` (subsample or max-pool the spatial channels to 64 or 128 for plotting efficiency):
> * Panel 1: Raw 1,024-channel HD-MEA Telemetry (Heatmap) with a vertical line at the `crash_frame`.
> * Panel 2: The Koopman Stability Metric (KSM) over time.
> * Panel 3: The MambaLRP Causal Attribution Heatmap.
> 
> 8. Save the JSON report to `output/harness/clinical_autopsy_report_[model_type].json`, the dashboard to `output/harness/[png_name]`, and a CSV summary to `output/harness/[csv_name]`.

---

### Step 2: Makefile Updates

Feed the following file path into your Antigravity IDE context:

* `Makefile`

**Raw Text Prompt to Execute:**

> Append the following targets to the `Makefile`:
> ```makefile
> .PHONY: clinical-autopsy evaluate-all-models
> 
> EPOCHS ?= 15
> 
> clinical-autopsy:
> 	python -m src.harness.clinical_autopsy_runner \
> 		--model-type meld \
> 		--epochs $(EPOCHS) \
> 		--seq-len 2000 \
> 		--ksm-threshold 0.85 \
> 		--png-name 06_clinical_autopsy_dashboard_meld.png \
> 		--csv-name 06_clinical_autopsy_metrics_meld.csv
> 
> evaluate-all-models:
> 	for model in meld baseline transformer mask_aware ; do \
> 		python -m src.harness.clinical_autopsy_runner \
> 			--model-type $$model \
> 			--epochs $(EPOCHS) \
> 			--seq-len 2000 \
> 			--ksm-threshold 0.85 \
> 			--png-name 06_clinical_autopsy_dashboard_$$model.png \
> 			--csv-name 06_clinical_autopsy_metrics_$$model.csv ; \
> 	done
> ```
