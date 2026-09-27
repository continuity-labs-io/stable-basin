Context: I am working on the `stable-basin` repository to build a diagnostic pipeline for the robust mouse rejuvenation (RMR) project. We have rough versions of the continuous-time physics metrics (thermodynamic and spectral) in place. Now, we need the evaluation harness to run these metrics over sliding windows and plot the physical divergence between aging biological models and ground-truth youthful baselines.

Task: Extract the sliding-window evaluation loop and plotting utilities from the `brain-gen` codebase (specifically parts of `brain_gen/eval/eval_runner.py`, `brain_gen/eval/rollout_sliding_windows.py`, and `brain_gen/eval/plotting.py`).

Destination: Port this logic into a new runner class called `ClinicalDiagnosticRunner` located at `src/harness/clinical_diagnostic_runner.py`.

Requirements:
1. Isolate the logic that chunks a continuous time-series into sliding windows, applies the metrics from `time_domain.py` and `spectral.py`, and aggregates the results across the time horizon.
2. Port the basic plotting scaffolding to generate the "rollout_divergence" curves and the sliding-window metric envelopes. 
3. Crucially, aggressively strip out all dependencies, plotting functions, JSON dumps, or logic related to `token_summary`, discrete codebooks, or language model evaluation. This harness is strictly for evaluating continuous physical dynamics (e.g., arrays of shape `[batch, channels, time]`).
4. Execution over perfection: Do not worry about perfect matplotlib styling, perfect memory management, or edge-case handling. The goal is purely to get a functional v1 scaffolding in place that iterates over the model outputs, calls the metrics, and plots the divergence curves. We will refine the visualization aesthetics later.
5. Include standard, calm standard-library logging (e.g., `logger.info("Running sliding-window diagnostic evaluation over cohort trajectory.")`).


