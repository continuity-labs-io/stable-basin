**Context Files to Load / Modify:**
* `src/echo/benchmarks/06_worm_gait_decline.py`

**Task: Paper 1 Rigor - Full-Series Thermodynamic Statistics**
We must move beyond visually eyeballing truncated plots. To make the "Waddington Basin Flattening" hypothesis publishable, we must compute robust statistical metrics (Mean/Std, KS-Statistic, Wasserstein Distance, and Cohen's d) over the *entire* evaluation time series, safely chunked to avoid Out-Of-Memory (OOM) errors.

**Core Objectives:**

**1. Safe Full-Trace Computation:**
* Add `scipy.stats` to the imports (for `ks_2samp` and `wasserstein_distance`).
* Add the `compute_full_trace(tracker, states, batch_size=1000)` helper function (as suggested) to safely chunk the curvature computation over the full length of the `macro_states` array without blowing out VRAM.
* Replace the `[::10][:1000]` striding in `main()` with this new chunking function to compute the complete traces for `trace_young_A`, `trace_old_A`, `trace_young_B`, and `trace_old_B`.

**2. Compute & Log Statistical Metrics:**
* For both Run A (GaussianEBM) and Run B (PrecisionWeightedEBM), calculate:
  * Mean curvature ($\mu$) and Standard Deviation ($\sigma$) for both Young and Old conditions.
  * The **Kolmogorov-Smirnov (KS) statistic** and p-value between the Young and Old traces.
  * The **Wasserstein Distance** between the Young and Old traces.
  * **Cohen's d (Effect Size):** Create a small helper to compute this: `(mean_young - mean_old) / pooled_std`.
* Log these metrics clearly using `logger.info`.

**3. Serialize Metrics to JSON:**
* Write all computed metrics to a structured dictionary.
* Save this dictionary to `output/echo/benchmarks/06_worm_gait_metrics.json` using the standard `json` module. 
* **Critical:** Convert all JAX/NumPy numeric types to standard Python `float()` before inserting them into the dictionary, otherwise `json.dump` will crash with a TypeError.

**4. Update Plotting:**
* Pass the full computed traces to `plot_ablation_results()`. The histograms (Panel C) and Laplace Flatline (Panel B) should now represent the entire evaluation dataset, not just a subset.

**Constraints:**
* Maintain Equinox and JAX functional purity.
* Keep logs professional and precise.
