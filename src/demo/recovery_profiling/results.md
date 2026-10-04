# Diazepam Recovery Profiling Results

This directory contains one-off diagnostic scripts and experiments evaluating the efficacy of various metrics (standard and thermodynamic) in tracking dose-dependent responses to Diazepam using the `PharmacologicalShockDataset`. 

### Experimental Sequence & Findings

1. **`00_screen_diazepam.py`**: 
   * **Goal**: Screen 10 proposed metrics (mean firing rate, burst rate, synchrony, recovery time via Autocorrelation/DMD, variance, PSD, PLV, PAC, MOU entropy) to see if they exhibited significant variance between control and drug segments.
   * **Result**: Only `Tau(AC)` (Recovery Time from lag-1 autocorrelation) passed the baseline variance gate. Most thermodynamic metrics (like MOU Entropy) were ruled out due to high baseline variance.

2. **`01_dose_response_tau.py`**:
   * **Goal**: Test if the `Tau(AC)` metric demonstrated a robust, dose-dependent curve. Evaluated over Control, 3uM, 10uM, 30uM, and 50uM conditions.
   * **Result**: `Tau(AC)` passed the dose gate. It consistently dropped at every step up in dose, and the change was robust against various parameter sweeps (MAD thresholds from -4.5 to -5.5 and bins from 5ms to 20ms).

3. **`02_time_drift_tau.py`**:
   * **Goal**: Test the "Time Gate" to ensure the observed `Tau(AC)` drop was a true dose-response and not just baseline temporal drift. The 5 recordings were collected sequentially over a 5.5-hour session.
   * **Result**: **FAILED**. The baseline drift slope observed in the 3-minute control recording extrapolated to a predicted drift of ~0.61 over 5.5 hours. The observed "drug" drop from Control to 50uM was only ~0.022. Because the predicted time drift dwarfed the observed signal by nearly 30x, the dose response was ruled out. The apparent dose effect is indistinguishable from standard organoid degradation/time drift.

### Conclusion
`Tau(AC)` and all other tested metrics have been ruled out for these continuous-recording organoid cohorts due to an inability to decouple the signal from baseline time drift.
