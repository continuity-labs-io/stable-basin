Context: I am building better measurement metrics for a robust mouse rejuvenation (RMR) project. 
I want to adapt continuous-time physics evaluation metrics from the `brain-gen` codebase to measure 
biological aging as a loss of system complexity.

Task: Extract the thermodynamic metrics (specifically Detrended Fluctuation Analysis (DFA), 
the Hurst exponent, and sliding-window variance/kurtosis for detecting critical slowing down) 
from `brain_gen/eval/rollout_metrics.py` and `rollout_sliding_windows.py`.
see: https://github.com/ricsinaruto/brain-gen


Destination: Port this logic into a new class called `TimeSeriesStabilityMetrics` located in `src/metrics/time_domain.py`.

Requirements:
1. Isolate the core mathematical logic for DFA, Hurst, and the sliding window statistical features.
2. Strip out *everything* related to discrete tokens, vector quantization (RVQ), codebooks, or language modeling. The methods should only accept standard continuous time-series numpy arrays (e.g., shape `[batch, channels, time]`).
3. Execution over perfection: Do not worry about extreme numeric accuracy, mathematical edge cases, NaN handling, or perfect coding fidelity right now. The sole goal is to get a rough, functional v1 baseline integrated into the repository so the structure is in place. We will fix the gory details later.
4. Add basic, calm logging using the standard library (e.g., `logger.info("Computing DFA for aging complexity metric.")`).
