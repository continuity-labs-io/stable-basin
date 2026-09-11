# Design Doc: Entropy Production as an EEG Marker of Brain State and Age

**Project:** Stable Basin, EEG track
**Status:** Draft v0.1 (September 11, 2026)
**Scope:** Stage 1 (Calibrate), Stage 2 (Aging), Stage 3 (Nonlinear, run as an experimental condition inside Stages 1 and 2)

---

## 0. Summary

We will measure **entropy production (EP)**, the degree to which brain dynamics are irreversible in time, from scalp EEG. The work answers two questions in order:

| Stage | Question | Role |
|---|---|---|
| **1. Calibrate** | Does our pipeline reproduce the known result that EP drops from wakefulness to deep sleep (N3)? | Validation. If this fails, stop and debug before Stage 2. |
| **2. Aging** | Does EP differ between young and older adults, and does it add information about age beyond standard spectral features? | Discovery. |
| **3. Nonlinear** | Do nonlinear estimators (a trained NESS energy-based model, or a model-free classifier) capture irreversibility that the linear estimators miss? | Experimental condition applied to Stages 1 and 2, not a separate study. |

Everything is preceded by **Stage 0**: proving on synthetic data with known ground truth that the estimators recover the correct EP. This is cheap and prevents weeks of debugging real data with a broken ruler.

Two open questions from the planning discussion are resolved here:

- **Which dataset?** (Section 4) Sleep-EDF is kept only as a two-channel smoke test. The multichannel calibration uses ANPHY-Sleep (83 channels). Aging uses MPI-LEMON, with the Dortmund Vital Study as replication.
- **Is the linear measure missing irreversibility?** (Section 9) A model-free classifier and a trained NESS model are run against the linear estimates, with surrogates that separate Gaussian from non-Gaussian irreversibility.

---

## 1. Background

A living system maintains itself away from thermodynamic equilibrium, so its dynamics are **irreversible**: a movie of it played backward is statistically distinguishable from the movie played forward. EP quantifies that distinguishability. In the NESS (non-equilibrium steady state) framework that Stable Basin is built on, EP is carried entirely by the **solenoidal flow** $Q$, the rotational component of the dynamics that the repo's `SolenoidalFlow` module already represents.

Prior evidence that EP tracks brain state:

- Human fMRI: brain dynamics break detailed balance, and entropy production rises with task demands (Lynn et al., 2021).
- Human fMRI with a linear stochastic model of exactly the form in Section 2.3: EP decreases with sleep depth, and regional irreversibility can be read directly off $Q$ (arXiv:2207.05197).
- Primate ECoG with a model-free classifier: neural dynamics become more time-reversible in deep sleep and ketamine anesthesia (de la Fuente et al., 2023).

What is missing: a careful **scalp EEG** replication of the sleep result, and any serious test of **EP versus age**. Studies of ordinary signal entropy versus age disagree (increases, decreases, and null results are all reported). EP is a different quantity, and it has not been well studied across the lifespan as far as a quick literature search shows.

---

## 2. Definitions

### 2.1 General definition

For a stationary process with trajectory $x_{0:\tau}$ and time-reversed trajectory $\tilde{x}_{0:\tau}$:

$$
\Phi \;=\; \lim_{\tau\to\infty}\frac{1}{\tau}\, D_{\mathrm{KL}}\!\left(P[x_{0:\tau}] \,\big\|\, P[\tilde{x}_{0:\tau}]\right) \quad \text{[nats / second]}
$$

$\Phi = 0$ if and only if the process is time-reversible (detailed balance). $\Phi > 0$ measures how quickly an observer accumulates evidence about the direction of time.

### 2.2 In the Stable Basin SDE

The repo's `Thermostat` integrates

$$
dx = -(\Gamma + Q)\,\nabla E(x)\,dt + \sqrt{2T}\,L\,dW, \qquad LL^\top = \Gamma,\;\; Q^\top = -Q
$$

with stationary density $p(x) \propto e^{-E(x)/T}$ (for constant $\Gamma$, $Q$). The dissipative part is reversible; only the solenoidal part produces entropy:

$$
\boxed{\;\Phi \;=\; \frac{1}{T}\; \mathbb{E}_{p}\!\left[\,(Q\nabla E)^\top \,\Gamma^{-1}\,(Q\nabla E)\,\right]\;}
$$

$\Phi = 0$ exactly when $Q = 0$. For a trained energy-based model this is a Monte Carlo average over samples, a few lines of JAX.

### 2.3 The linear special case (multivariate Ornstein–Uhlenbeck)

Restrict the energy to a quadratic, $E = \tfrac12 x^\top \Sigma^{-1} x$ with $T = 1$. The drift becomes linear:

$$
dx = A\,x\,dt + \text{noise}, \qquad A = -(\Gamma + Q)\,\Sigma^{-1}
$$

This is a multivariate Ornstein–Uhlenbeck (MOU) process, the continuous-time version of a first-order vector autoregression. Every quantity follows from two covariance matrices:

- $\Sigma = \mathrm{Cov}(x)$
- $C(\tau) = \mathbb{E}[x_{t+\tau}\,x_t^\top] = e^{A\tau}\,\Sigma$
- $A = \tfrac{1}{\tau}\log\!\big(C(\tau)\,\Sigma^{-1}\big)$
- $\Gamma = -\tfrac12\,(A\Sigma + \Sigma A^\top)$ and $Q = -\tfrac12\,(A\Sigma - \Sigma A^\top)$

$$
\boxed{\;\Phi_{\text{MOU}} \;=\; -\operatorname{Tr}\!\left(\Gamma^{-1}\, Q\, \Sigma^{-1}\, Q\right) \;\ge\; 0\;}
$$

**Ground-truth test case** (used in Stage 0): with $\Sigma = \Gamma = I_2$ and $Q = \begin{pmatrix}0 & q\\ -q & 0\end{pmatrix}$, the exact answer is $\Phi = 2q^2$. All three linear estimators below were checked numerically against this case while writing this doc.

### 2.4 The three linear estimators

All three assume Gaussian statistics. They differ in what else they assume and what they return.

**Estimator A: lag-τ pairwise (simplest).** Compare the joint distribution of $(x_t, x_{t+\tau})$ with its reversal:

$$
S_f = \begin{pmatrix}\Sigma & C^\top \\ C & \Sigma\end{pmatrix},\quad
S_b = \begin{pmatrix}\Sigma & C \\ C^\top & \Sigma\end{pmatrix},\quad
\hat\Phi_A(\tau) = \frac{\tfrac12\operatorname{Tr}(S_b^{-1}S_f) - k}{\tau}
$$

where $k$ is the number of channels or components. This is a lower bound that converges to the true EP of an MOU process as $\tau \to 0$ (numerically: 56%, 90%, 98% of the true value at $\tau = 0.5, 0.1, 0.02$ time constants). It needs no matrix logarithm and is the sanity check for the other two.

**Estimator B: MOU fit (interpretable).** Compute $A$, $\Gamma$, $Q$ as in Section 2.3, then $\Phi_{\text{MOU}}$. This is the estimator that matches the repo's thermostat, and it returns the matrix $Q$ itself. Nodal irreversibility $\sum_j |Q_{ij}|$ gives a topographic map of where irreversibility lives.

Diagnostics that must pass per epoch: the matrix logarithm must be real (choose $\tau$ well below half the period of the fastest retained oscillation, e.g., 1–2 samples at 100–250 Hz); $\hat\Gamma$ must be positive definite. If $\hat\Gamma$ is not positive definite, the first-order Markov model is misspecified for that epoch; log it, do not silently clip it.

**Estimator C: spectral (recommended primary).** For a stationary Gaussian process, EP is determined entirely by the cross-spectral matrix $S(f)$. Define the **EP spectral density**:

$$
\varphi(f) = \operatorname{Tr}\!\left(S(f)^{-\top}\,S(f)\right) - k, \qquad
\Phi_C = \int_0^{f_s/2} \varphi(f)\,df \quad \text{[nats/s]}
$$

Why it is the primary estimator:

- **No Markov assumption.** EEG is not a first-order Markov process in sensor space, so Estimator B is always somewhat misspecified. Estimator C is the exact EP of the Gaussian process with the observed spectra.
- **Band-resolved.** $\Phi_{\text{band}} = \int_{\text{band}} \varphi(f)\,df$ says *which rhythms* carry irreversibility (alpha? slow waves?). This connects EP directly to the repo's existing spectral metrics.
- **Intuitive.** $\varphi(f) = 0$ at any frequency where the imaginary part of the cross-spectrum vanishes. Irreversibility in Gaussian signals comes only from consistent phase lags between channels.

Estimate $S(f)$ by Welch's method with enough segments per epoch (at least several times $k$) and mild shrinkage. The estimate is biased upward by finite data; the surrogate null (Section 5.5) calibrates this.

### 2.5 Invariances and bounds

These properties drive the preprocessing rules in Section 5.

| Operation | Effect on EP | Consequence |
|---|---|---|
| Invertible instantaneous mixing (re-referencing, ICA unmixing, volume conduction) | **Unchanged** | Volume conduction cannot create EP. ICA is safe. |
| Dropping channels or components, projecting to fewer dimensions | **Can only decrease** | PCA gives a lower bound. Keep $k$ fixed across all comparisons. |
| Coarser sampling in time | **Can only decrease** | Downsampling gives a lower bound. |
| Adding independent, time-reversible noise | **Can only decrease** | Low SNR dilutes EP. SNR must be a covariate. |
| **The same** linear filter applied to every channel (causal *or* zero-phase) | **Gaussian EP density unchanged** inside the passband; removed outside it | Filtering only selects bands. |
| **Different** filters or delays on different channels | **Can create spurious EP** | Never process channels differently. |
| Adding *time-irreversible* artifacts (blinks, ECG, muscle bursts) | **Can increase** | The main artifact risk. See Section 10. |

> **Correction to earlier advice.** In the planning discussion I said zero-phase (forward-backward) filtering "erases" irreversibility and that only causal filters should be used. That was wrong for this offline analysis. Applying the same filter to every channel multiplies the cross-spectrum by $|H(f)|^2$, which cancels in $\varphi(f)$; this was verified numerically. Zero-phase filtering is a problem for *real-time prediction*, because it uses future samples, which is the issue in the phase-prediction paper. For EP, the real rule is: identical processing on every channel. For the nonlinear stage, prefer zero-phase or linear-phase filters, because nonlinear-phase (e.g., IIR causal) filters distort the higher-order phase relationships that carry non-Gaussian irreversibility.

### 2.6 What EP here does and does not mean

EP estimated from EEG is the time-irreversibility of a **coarse-grained, noisy, low-dimensional observable** of the brain. Every bound in Section 2.5 points the same way: the measured value is a **lower bound** on the irreversibility of the underlying dynamics. It is an information-theoretic quantity in nats per second. It is **not** the brain's metabolic heat dissipation, and it must not be described as such in any write-up.

---

## 3. Questions and hypotheses

**Q1 (Calibrate).** Within subjects, is EP lower in N3 than in wakefulness?

- **H1:** $\Phi(\text{Wake}) > \Phi(\text{N3})$, paired within subject.
- Secondary, exploratory: a monotone ordering Wake > N1 > N2 > N3; the position of REM (no strong prior).
- Scientific risk: slow waves in N3 are **traveling waves** that propagate across the scalp with consistent lags, which could *raise* irreversibility in the slow-wave band even if it falls overall. The band-resolved $\varphi(f)$ is designed to catch this. A result like "EP falls in alpha/beta but rises in the delta band" would be informative, not a failure.

**Q2 (Aging).** Does resting-state EP differ between young and older adults, and does it carry age information beyond standard spectral features?

- **H2a:** $\Phi$ differs between age groups. Tested **two-sided**, because the direction is not established.
- **H2b:** Adding EP features to a model with standard spectral features (band powers, individual alpha frequency, aperiodic exponent) improves cross-validated age prediction.
- H2b is the real test of "EP as an age marker." A group difference that disappears once alpha slowing and 1/f changes are accounted for would mean EP is a re-description of known effects.

**Q3 (Nonlinear condition).** Within Q1 and Q2, do nonlinear estimators find irreversibility beyond the Gaussian estimate, and do they sharpen the sleep or age effects? Decision rule in Section 9.4.

**Pre-registration.** Before touching Stage 2 data, write down one primary outcome: broadband $\Phi_C$ (Estimator C), $k = 8$ components, eyes-closed rest, 1–45 Hz. Everything else is labeled exploratory and corrected for multiple comparisons (FDR).

---

## 4. Datasets (Open Question 1: which dataset?)

### 4.1 Candidates

| Dataset | Channels | Subjects / ages | Why | Limitations |
|---|---|---|---|---|
| **Sleep-EDF Expanded, Sleep Cassette** (PhysioNet) | 2 EEG (Fpz-Cz, Pz-Oz), 100 Hz | 78 healthy adults, **ages 25–101**, mostly 2 nights each | Standard, small, MNE has a fetcher (`mne.datasets.sleep_physionet.age`). Wide age range enables a sleep × age analysis. | With 2 channels, $Q$ is a single number. Old cassette hardware, 100 Hz, R&K scoring (merge S3+S4 into N3). |
| **ANPHY-Sleep** (OSF, doi:10.17605/OSF.IO/R26FH) | **83 EEG** + EOG, EMG, ECG | 29 healthy adults, mean age about 32 | High-density, modern, includes ECG and EOG for artifact control, AASM scoring. | Small N; young only; large files. |
| **MPI-LEMON** | 62 channels, 2500 Hz raw | 227 healthy adults: **153 young (20–35)**, **74 older (59–77)** | Clean two-group design, 16 min rest alternating 1-min eyes-open / eyes-closed blocks, rich covariates (MRI, cognition, blood). Widely used for EEG brain-age work. | Gap in ages 36–58. Sex imbalance: 29% female in the young group versus 50% in the older group. A few raw files reportedly corrupted. |
| **Dortmund Vital Study** (Getzmann et al., 2024) | 64 channels | **608 adults, ages 20–70** (continuous), rest before and after a 2-h cognitive session; **~5-year follow-up in 208** | Continuous age, large N, and **longitudinal within-person change**. | Use pre-task rest to avoid fatigue effects (or model them). |

### 4.2 Decision

- **Stage 1 smoke test: Sleep-EDF.** Fast, tiny, well-trodden. It proves the plumbing end to end with $k = 2$. It is *not* the calibration result.
- **Stage 1 calibration: ANPHY-Sleep.** Multichannel, modern, with ECG and EOG for artifact control. This resolves the two-channel caution.
- **Stage 2 primary: MPI-LEMON.** Two clean age groups for H2a/H2b.
- **Stage 2 replication: Dortmund Vital Study.** Continuous age for a dose-response curve, and the longitudinal subset for within-person change: the "velocity of aging" framing.
- **Stage 2 bonus: Sleep-EDF across ages 25–101.** Does N3 EP change with age? Cheap once Stage 1 works.

Before committing: confirm current access terms and download formats for ANPHY (OSF) and Dortmund (OpenNeuro / Sci Data record), and count usable LEMON raw files.

---

## 5. Pipeline

### 5.1 Data flow

```
raw recording (EDF / BrainVision / BIDS)
  → load + channel metadata                    [src/data/eeg/<dataset>.py]
  → identical filtering, notch, resample       [src/eeg/preprocess.py]
  → artifact handling (ICA: ocular, cardiac)   [src/eeg/preprocess.py]
  → epoching by stage / condition              [src/eeg/epochs.py]
  → reference + PCA to k components            [src/eeg/reduce.py]
  → estimators A, B, C  (+ D, E in Stage 3)    [src/metrics/entropy_production.py]
  → surrogate nulls                            [src/metrics/ep_surrogates.py]
  → per-subject tables (parquet)               [results/<stage>/<dataset>/]
  → statistics + figures                       [experiments/<stage>_*.py]
```

### 5.2 Preprocessing rules

1. **Start from raw data** where available (LEMON raw at 2500 Hz rather than the preprocessed release), so every step is known and identical across groups.
2. **Filter identically on every channel.** High-pass around 0.5 Hz (sleep) or 1 Hz (rest), low-pass 45 Hz, line-noise notch. Zero-phase or linear-phase FIR (Section 2.5). Band selection for scientific questions is done later with $\varphi(f)$, not by re-filtering.
3. **Resample** to 100 Hz (sleep) or 250 Hz (rest), with the same anti-alias filter on every channel.
4. **Artifacts.** Blinks, saccades, and heartbeats are strongly time-asymmetric and will masquerade as brain irreversibility.
   - Remove ocular and cardiac components with ICA, using EOG and ECG channels where present (ANPHY has both).
   - Reject epochs with residual high EOG/EMG power, using the same thresholds across conditions and groups.
   - Record per-epoch artifact indices (EOG power, high-frequency EMG power, residual ECG correlation) as covariates.
5. **Reference** to the average (this makes the data rank $d-1$; handled by the PCA step).
6. **Dimensionality.** PCA to $k$ components fitted on pooled data within a dataset, so every subject lives in the same component space. Primary $k = 8$; sensitivity $k \in \{4, 8, 16\}$. Sleep-EDF is fixed at $k = 2$.
7. **Epochs.** Sleep: 30-s epochs matching the scoring, excluding epochs adjacent to a stage change. Rest: 20-s windows inside each 1-min eyes-open/eyes-closed block, dropping the first 5 s after each instruction.
8. **Equalize data.** Estimator bias depends on data length and $k$. Use the same number of epochs per subject per condition (subsample to the minimum, or report EP as a function of data length).

### 5.3 Estimator module API

```python
# src/metrics/entropy_production.py  (NumPy/SciPy; no deep-learning dependency)

def ep_pairwise(x, fs, lag) -> float                  # Estimator A, nats/s
def ep_mou(x, fs, lag) -> MOUResult                   # Estimator B: phi, A, Gamma, Q, diagnostics
def ep_spectral(x, fs, nperseg, fmax=None) -> SpectralEP  # Estimator C: phi_total, freqs, phi_f
def band_ep(spec: SpectralEP, band: tuple) -> float

# src/metrics/ep_surrogates.py
def reversible_gaussian_surrogate(x, fs, rng) -> ndarray   # spectrum Re S(f): same power, same real coherence, zero imaginary coherence
def phase_randomized_surrogate(x, rng) -> ndarray          # multivariate phase randomization: keeps full S(f), destroys non-Gaussian structure
def reversible_mou_surrogate(mou: MOUResult, n, fs, rng) -> ndarray  # simulate fitted model with Q = 0
```

`x` is always `(n_samples, k)`. All functions are pure and unit-tested against Section 2.3's closed form.

### 5.4 Per-epoch outputs

For every epoch: $\hat\Phi_A$ at 3 lags, $\hat\Phi_B$, $\hat\Phi_C$, $\varphi(f)$ on a fixed frequency grid, $\hat Q$ (from B), the surrogate null distribution summary, plus covariates: total power, band powers, individual alpha frequency, aperiodic exponent (specparam/FOOOF), artifact indices, and decorrelation time.

### 5.5 Null model and bias correction

Finite data always gives a positive EP estimate, even for a reversible process. So every reported value is compared against a surrogate with the same length, $k$, and spectra, but with irreversibility removed:

- **Primary null: reversible Gaussian surrogate.** Replace $S(f)$ by its real part, $\operatorname{Re} S(f)$ (still a valid spectral matrix), and synthesize data from it via a per-frequency Cholesky factor applied to complex white noise. Same power spectra, same real coherence, zero EP by construction.
- Report **excess EP** $= \hat\Phi - \operatorname{median}(\hat\Phi_{\text{null}})$ and the percentile of $\hat\Phi$ in the null distribution (200 surrogates per subject and condition is enough).

### 5.6 Statistics

- **Stage 1:** per-subject median EP per stage. Primary test: paired Wilcoxon, Wake versus N3. Model: linear mixed model $\log\Phi \sim \text{stage} + (1\,|\,\text{subject})$, with night as a nuisance factor in Sleep-EDF. Report the fraction of subjects showing Wake > N3.
- **Stage 2:** see Section 8.3.

---

## 6. Stage 0: Prove the ruler works (synthetic ground truth)

Run before any real data. All of it lives in `tests/test_entropy_production.py` and runs in CI.

1. **Closed-form MOU.** $\Sigma = \Gamma = I$, $Q$ as in Section 2.3. Assert all three estimators recover $\Phi = 2q^2$ within tolerance, for several $q$ including $q = 0$.
2. **Random MOU systems** with $k \in \{2, 8, 16\}$, including oscillatory modes near 10 Hz, simulated at realistic sampling rates. Check agreement and bias as a function of epoch length. Output: a lookup table "minimum epoch length for bias < 10% at dimension $k$."
3. **Invariance tests.** Random invertible mixing: EP unchanged. Identical zero-phase filter on all channels: $\varphi(f)$ unchanged in the passband. A 5 ms delay on one channel only: EP rises (this confirms the detector of processing bugs works).
4. **Null calibration.** With $Q = 0$, the surrogate-corrected excess EP should be centered at zero with a false-positive rate near 5%.
5. **Artifact sensitivity.** Inject synthetic blink waveforms (fast rise, slow decay) and ECG-like complexes into a reversible process. Measure how much spurious EP they create before and after the ICA step. This sets the rejection thresholds used in Section 5.2.
6. **SNR dilution curve.** Add white noise at decreasing SNR to an irreversible process. This quantifies how much of a group difference could be explained by an SNR difference alone (needed for Stage 2).

**Exit criterion:** all tests pass; the bias table and dilution curve exist as figures.

---

## 7. Stage 1: Calibrate (sleep)

### 7.1 Stage 1a: Smoke test on Sleep-EDF

- 10 subjects, both nights, $k = 2$ (Fpz-Cz, Pz-Oz). Estimators A, B, C.
- Goal is plumbing, not science: loading, stage alignment, epoching, estimators, nulls, and the results table all work end to end.
- Sanity outputs: EP distributions per stage, null distributions, runtime per subject.

### 7.2 Stage 1b: Calibration on ANPHY-Sleep

- All 29 subjects; $k \in \{4, 8, 16\}$; estimators A, B, C with surrogate nulls.
- Stages: Wake (within the sleep period and the pre-sleep period, analyzed separately), N1, N2, N3, REM.
- Artifact covariates per epoch; ECG-based cardiac ICA removal.

### 7.3 Success and stop criteria

**Pass (proceed to Stage 2) if all hold for the primary estimator C at $k = 8$:**

1. Waking EP is above the surrogate null (EEG irreversibility is detectable at all).
2. Wake > N3 in the paired test ($p < 0.05$), with the direction holding in at least 70% of subjects.
3. The direction survives adding artifact covariates (EMG and EOG power), because wake has more muscle and eye activity than N3 and that confound favors H1.
4. The direction agrees across estimators A, B, C and across $k$.

**Stop and debug if:**

- Waking EP is not above the null: sensitivity problem (too little data per epoch, $k$ too small, or over-aggressive filtering).
- N3 > Wake broadband: inspect band-resolved $\varphi(f)$ for a traveling-slow-wave effect, then check artifacts, before concluding anything.

### 7.4 Stage 1 figures

1. EP by sleep stage (per-subject lines, group summary), with the null band.
2. $\varphi(f)$ for Wake versus N3: which frequencies carry the difference.
3. Nodal irreversibility topographies (from $\hat Q$) for Wake versus N3.
4. Robustness grid: estimator × $k$ × lag.

---

## 8. Stage 2: Aging

### 8.1 Protocol (MPI-LEMON)

- Eyes-closed rest (primary) and eyes-open rest (secondary), analyzed separately, never pooled.
- Same pipeline as Stage 1, frozen at the Stage 1 settings, with only the epoching changed (Section 5.2, rule 7).
- Pre-register the primary outcome (Section 3) before computing any group statistics.

### 8.2 Confounds and controls

Age changes many EEG properties at once. EP is only interesting if it is not just a re-description of them.

| Confound | Why it matters | Control |
|---|---|---|
| **Slowing of rhythms** (lower individual alpha frequency with age) | EP in nats/second scales with how fast the dynamics run. Slower dynamics lower EP per second even with identical "shape." | Report **time-rescaled EP** (EP per decorrelation time, and per alpha cycle) alongside EP per second. |
| **Aperiodic (1/f) changes** (flatter spectra with age) | Changes the spectral weighting of $\varphi(f)$. | Aperiodic exponent as a covariate; band-resolved analysis. |
| **Signal-to-noise ratio** | Lower SNR dilutes EP (Section 2.5), which could fake an age effect. | Total power and SNR proxies as covariates; the Stage 0 dilution curve bounds the effect SNR alone could produce. |
| **Vigilance** | Drowsiness during eyes-closed rest changes dynamics. | Vigilance proxy per epoch (e.g., alpha/theta ratio); eyes-open condition as a check. |
| **Artifacts** | Eye movements and muscle tension may differ by age. | Artifact indices as covariates; identical rejection thresholds. |
| **Sex imbalance** | LEMON's older group has a much higher proportion of women. | Sex as a covariate; sex-stratified sensitivity analysis. |
| **Data quantity** | Estimator bias depends on data length. | Equal number of clean epochs per subject. |

### 8.3 Analyses

**H2a (group difference).** Two-sided permutation test on subject-level excess EP; ANCOVA with the covariates above. Report effect sizes with confidence intervals, not just p-values.

**H2b (incremental value).** Nested, cross-validated models predicting age group (LEMON) or age in years (Dortmund):

- Baseline features: log band powers, individual alpha frequency, aperiodic exponent and offset.
- Baseline + EP features: broadband $\Phi_C$, band-resolved EP (delta, theta, alpha, beta), and a low-dimensional summary of the nodal irreversibility map.
- Metric: AUC (groups) or mean absolute error (years). The improvement's confidence interval comes from repeated cross-validation, and its null from permuting the EP features only.

### 8.4 Replication and extensions

- **Dortmund Vital Study:** pre-task rest; EP versus continuous age (linear and quadratic terms). Longitudinal subset: does within-person EP change over about 5 years, and does the change relate to baseline age?
- **Sleep-EDF across ages 25–101:** does N3 EP change with age? This joins Stages 1 and 2 in one analysis.

### 8.5 Success criteria

- **Strong result:** H2a significant *and* H2b shows a reliable improvement over the spectral baseline, replicated in Dortmund.
- **Moderate result:** H2a significant but H2b not: EP tracks age, but through known spectral changes. Still reportable.
- **Null result:** no group difference after controls. Report it; EP is then a state marker (Stage 1) but not an age marker.

---

## 9. Stage 3: Nonlinear condition (Open Question 2: is the linear measure missing irreversibility?)

The linear estimators capture only irreversibility carried by phase lags between channels (second-order statistics). Real EEG also has **waveform asymmetry**: sharp rises and slow decays, sawtooth-like shapes within a single channel. That is non-Gaussian irreversibility, invisible to Estimators A–C. Stage 3 asks how much of it there is and whether it matters.

### 9.1 Estimator D: model-free classifier

Train a network to tell forward windows from time-reversed windows of the $k$-dimensional component signals (window length about 0.5–2 s). Two established variants:

- **Arrow-of-time classifier** (Seif et al., 2021): the trained classifier's log-odds estimates the EP of each window. This is the approach used on primate ECoG in de la Fuente et al. (2023).
- **NEEP** (Kim et al., 2020): an explicitly antisymmetric network trained with a variational objective whose optimum equals the EP per step. Gives a lower bound directly in nats.

Train with subject-wise cross-validation (never test on a subject seen in training). Output: per-epoch EP estimates comparable with A–C.

### 9.2 Estimator E: the fitted NESS energy-based model (the repo's Echo model)

Fit the full model of Section 2.2, $f_\theta(x) = -(\Gamma + Q)\nabla E_\theta(x)$ with noise $2T\Gamma$, to the $k$-dimensional components by maximizing the Euler–Maruyama transition likelihood at a coarse step $\tau$ (matching Estimator B's lag).

- **Nested comparison:** a quadratic $E_\theta$ reproduces the MOU model exactly, so the nonlinear model's held-out likelihood can be compared directly against Estimator B.
- **EP:** evaluate the Section 2.2 formula by Monte Carlo, once over samples from the fitted model's own stationary distribution (by running the thermalizer) and once over the data. Disagreement between the two flags a poor fit.
- **Ground-truth validation first:** add a known solenoidal term to the repo's Müller–Brown toy potential, where the true EP can be computed from the formula, and confirm the fit recovers it.

**Required repo changes before Estimator E is usable:**

1. **Fix the Markov-blanket masking.** Masking $\Gamma$ element-by-element can make it non-positive-definite. The drift then uses the masked matrix while the noise uses an eigenvalue-clipped version, which breaks the fluctuation–dissipation relationship, so the stationary density is no longer $e^{-E/T}$ and the EP formula no longer applies. For this study, drop the mask (the EEG application does not need a Markov-blanket partition), or parameterize $\Gamma$ so it is positive definite *within* the allowed pattern, and use the same matrix for drift and noise.
2. **Add a training loop.** The model is currently never fit to data.
3. **Remove what is not needed here:** the precision head and the hull's sensory degradation.
4. **Compute EP in float64.**

**Known limitation:** EEG is a noisy, filtered projection of hidden brain dynamics, not a directly observed diffusion. Fitting an SDE in component space is an approximation. The principled version is a **latent** NESS SDE with an observation model, trained variationally. That is future work, and the natural place for the repo's state-space-model code to re-enter.

### 9.3 Separating Gaussian from non-Gaussian irreversibility

Run Estimator D on three versions of each epoch:

| Input | Gaussian EP | Non-Gaussian EP | Expected classifier EP |
|---|---|---|---|
| (a) Real data | kept | kept | total |
| (b) Multivariate phase-randomized surrogate (keeps the full $S(f)$) | kept | removed | Gaussian part only |
| (c) Reversible Gaussian surrogate ($\operatorname{Re} S(f)$) | removed | removed | about 0 (checks the classifier's bias) |

Non-Gaussian irreversibility ≈ D(a) − D(b). Because classifier estimates are lower bounds and data-hungry, treat this as approximate and compare groups or stages, not absolute values.

### 9.4 Decision rule

Adopt the nonlinear estimators as the main measure only if at least two of these hold:

1. D(a) is reliably greater than D(b): non-Gaussian irreversibility exists.
2. Estimator E beats the MOU model on held-out likelihood.
3. A nonlinear estimator gives a larger standardized effect for Wake versus N3 or young versus old than $\Phi_C$.

If none hold, the finding is that EEG irreversibility is essentially Gaussian phase-lag structure, and the linear estimator is sufficient. That is a clean, publishable answer to Open Question 2.

**Compute:** Stages 0–2 run on a CPU. Stage 3's classifier and energy-model training is where the RTX 4090 earns its keep.

---

## 10. Risks and other concerns

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Eye artifacts:** blinks have a fast-closing, slow-opening shape that is strongly irreversible. | ICA with EOG reference; epoch rejection; EOG covariate; Stage 0 injection test sets thresholds. |
| 2 | **Cardiac artifact:** the heartbeat waveform is irreversible and periodic and projects onto many channels. | ICA with ECG reference; residual ECG correlation as a covariate. |
| 3 | **Muscle artifact:** higher in wake than in N3, which biases Stage 1 toward the expected result. | 45 Hz low-pass; EMG-based rejection; EMG covariate; pass criterion 3 in Section 7.3. |
| 4 | **Traveling slow waves** could raise delta-band EP in N3. | Band-resolved $\varphi(f)$; interpret by band. |
| 5 | **Finite-sample bias** grows with $k$ and shrinks with data length. | Fixed $k$; equal data per subject; surrogate nulls; Stage 0 bias table. |
| 6 | **Nonstationarity:** drifts and state changes inside an epoch look irreversible. | High-pass; exclude transition epochs; check variance stability within each epoch. |
| 7 | **Timescale confound:** faster dynamics mean more EP per second. | Time-rescaled EP alongside EP per second. |
| 8 | **SNR dilution** can fake group differences. | SNR covariates; Stage 0 dilution curve. |
| 9 | **Channel-specific processing** creates spurious lags. | One pipeline for every channel; Stage 0 single-channel-delay test. |
| 10 | **Forking paths:** many estimators, bands, $k$ values, conditions. | One pre-registered primary outcome; everything else exploratory with FDR. |
| 11 | **Over-interpretation:** calling EP "metabolic dissipation" or "a consciousness meter." | Section 2.6 language in all write-ups. |
| 12 | **Data access** terms or formats differ from expectations. | Verify all four datasets before Stage 1b. |

---

## 11. Repository integration

```
src/
  data/eeg/
    sleep_edf.py         # MNE fetcher, R&K → AASM stage mapping
    anphy.py             # OSF download, 83-ch montage, AASM annotations
    lemon.py             # raw BrainVision, EO/EC block parsing
    dortmund.py          # BIDS loader, pre/post task, follow-up
  eeg/
    preprocess.py        # identical filtering, resample, ICA (ocular, cardiac)
    epochs.py            # stage/condition epoching, transition exclusion
    reduce.py            # average reference, pooled PCA to k components
  metrics/
    entropy_production.py  # estimators A, B, C (+ band_ep)
    ep_surrogates.py       # reversible Gaussian, phase-randomized, reversible MOU
    ep_classifier.py       # Stage 3 estimator D (arrow-of-time / NEEP)
  echo/
    ...                    # Stage 3 estimator E: masking fix, training loop, EP in float64
experiments/
  stage1_sleep_smoke.py
  stage1_sleep_anphy.py
  stage2_aging_lemon.py
  stage2_aging_dortmund.py
  stage3_nonlinear.py
configs/ep/*.yaml          # one config per experiment; frozen after Stage 1
tests/test_entropy_production.py   # Stage 0
```

- **Dependencies:** `mne`, `mne-bids`, `numpy`, `scipy`, `pandas`, `pyarrow`, `statsmodels`, `scikit-learn`, `specparam`; `jax`/`equinox` for Stage 3 only.
- **Makefile targets:** `ep-tests`, `ep-sleep-smoke`, `ep-sleep`, `ep-aging`, `ep-nonlinear`.
- **Reproducibility:** fixed seeds; a data manifest with file checksums; results as parquet tables keyed by dataset, subject, epoch, estimator, and $k$; configs versioned with results.
- **Scope freeze:** everything outside this doc goes to `icebox/` until the Stage 1 go/no-go. In anything written for publication, use standard names (e.g., "DMD eigenvalue stability" rather than "Koopman Stability Metric").

---

## 12. Milestones

| Milestone | Deliverable | Gate |
|---|---|---|
| **M0** Estimators + Stage 0 | Tested `entropy_production.py`, bias table, dilution curve | All Stage 0 tests pass |
| **M1** Sleep-EDF smoke test | End-to-end results table for 10 subjects | Pipeline runs cleanly |
| **M2** ANPHY calibration | Stage 1 figures 1–4 | **Go/no-go** on Section 7.3 |
| **M3** Pre-registration + LEMON | Frozen config, H2a/H2b results | Pre-registration written before group statistics |
| **M4** Replication | Dortmund continuous-age and longitudinal results; Sleep-EDF sleep × age | — |
| **M5** Nonlinear condition | Section 9.3 table; Section 9.4 decision | — |
| **M6** Preprint | Methods (Stage 0), calibration, aging, nonlinear comparison | — |

---

## 13. Open decisions for the next session

1. Confirm Estimator C as the pre-registered primary, and the primary $k$ and band.
2. Verify access and formats for ANPHY-Sleep and the Dortmund Vital Study; count usable LEMON raw files.
3. Choose the Stage 3 classifier variant (arrow-of-time classifier versus NEEP) and window length.
4. Decide whether the Markov-blanket structure is dropped for Estimator E or re-parameterized.

---

## 14. References

- Lynn, C. W., Cornblath, E. J., Papadopoulos, L., Bertolero, M. A., & Bassett, D. S. (2021). Broken detailed balance and entropy production in the human brain. *PNAS*.
- *Entropy production of multivariate Ornstein–Uhlenbeck processes correlates with consciousness levels in the human brain.* arXiv:2207.05197.
- de la Fuente, L. A., et al. (2023). Temporal irreversibility of neural dynamics as a signature of consciousness. *Cerebral Cortex*, 33(5), 1856.
- Seif, A., Hafezi, M., & Jarzynski, C. (2021). Machine learning the thermodynamic arrow of time. *Nature Physics*.
- Kim, D.-K., Bae, Y., Lee, S., & Jeong, H. (2020). Learning entropy production via neural networks. *Physical Review Letters*.
- Kemp, B., et al. Sleep-EDF Database Expanded. PhysioNet. https://physionet.org/content/sleep-edfx/1.0.0/
- Wei, X., Avigdor, T., Ho, A., et al. (2024). ANPHY-Sleep: an open sleep database from healthy adults using high-density scalp electroencephalogram. *Scientific Data*, 11, 896.
- Babayan, A., et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. *Scientific Data*.
- Getzmann, S., Gajewski, P. D., Schneider, D., & Wascher, E. (2024). Resting-state EEG data before and after cognitive activity across the adult lifespan and a 5-year follow-up. *Scientific Data*.
- Engemann, D. A., et al. (2022). A reusable benchmark of brain-age prediction from M/EEG resting-state signals. *NeuroImage*.
- Voytek, B., et al. (2015). Age-related changes in 1/f neural electrophysiological noise. *Journal of Neuroscience*.
- Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical Review Letters*.
