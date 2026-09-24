# Implement Statistical Significance Tests for Baseline Metrics

## Context
Currently, the baseline biological metrics for the C. elegans worm gait dataset (Time Domain: AR(1) coefficient, State Variance; Spectral Domain: Peak Frequency) are output strictly as means and standard deviations (e.g., in `01_worm_gait_baseline_metrics.json`). We need to rigorously validate that the differences observed between the young (baseline) and old (degraded) cohorts are statistically significant.

## Objective
Implement formal statistical significance testing for the univariate baseline metrics. Specifically, compute p-values and effect sizes to validate the structural differences between cohorts using Welch's t-test and the Mann-Whitney U test.

## Execution Directives

1. **Target the Baseline Metrics Evaluator:**
   - Locate the script or pipeline step responsible for computing the baseline biological metrics (likely `src/benchmarks/worm_gait/01_worm_gait_baseline.py` or the corresponding metrics calculator module).
   - Extract the raw, unaggregated sample metric arrays for both the young and old cohorts (since you need the full sample distributions, not just the pre-calculated means and stds, to run the statistical tests).

2. **Implement Welch's t-test:**
   - For each metric (AR(1) coefficient, state variance, and peak frequency), compute Welch's t-test using `scipy.stats.ttest_ind(..., equal_var=False)`.
   - *Rationale:* Welch's t-test is strictly required over a standard Student's t-test because the empirical data exhibits severe heteroscedasticity (e.g., the state standard deviation dramatically increases from 0.133 in the young cohort to 0.576 in the old cohort).

3. **Implement the Mann-Whitney U Test (Wilcoxon Rank-Sum):**
   - For each metric, run the non-parametric Mann-Whitney U test using `scipy.stats.mannwhitneyu(...)`.
   - *Rationale:* This provides a robust fallback. Biological time-series metrics (especially expanding variances and shifting autoregressive coefficients) are likely to be non-Gaussian and heavily skewed, violating the core assumptions of t-tests.

4. **Compute Univariate Effect Sizes:**
   - Calculate Cohen's d for each metric to provide a standardized measure of the mean difference.
   - (Optional but recommended) Compute the rank-biserial correlation as a non-parametric effect size to pair with the Mann-Whitney U test.

5. **Update JSON Serialization:**
   - Modify the output serialization schema to append these new statistical results into `01_worm_gait_baseline_metrics.json` alongside the existing means and standard deviations.
   - Expected additions per metric:
     - `welch_t_stat` and `welch_p_value`
     - `mann_whitney_u_stat` and `mann_whitney_p_value`
     - `cohens_d`

6. **Integrate Results into the Paper via YAML Prompts:**
   - Locate the prompt responsible for generating the baseline results section in `paper/sharpening_the_tack/paper_metadata.yaml` (likely `ai_results_baseline` or similar).
   - Generically alter this prompt so that it explicitly instructs the AI to include and cite the computed significance tests (e.g., Welch's or Mann-Whitney p-values, and Cohen's d effect sizes) when reporting the means and variances for the time and spectral domains.
   - Inject the newly computed test results directly into the relevant paragraphs of the Results section in `paper/sharpening_the_tack/sharpening_the_tack.tex`.

7. **Testing & Constraints:**
   - **Do not remove or alter** the existing mean and standard deviation outputs; these new tests must be strictly additive.
   - Write a new unit test validating the statistical outputs, strictly adhering to the mandated `ARRANGE`, `ACT`, `ASSERT` block structure defined in `AGENTS.md`.
