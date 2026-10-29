# Recovery Ceiling — Implementation Design

Status: design, not yet implemented
---

## 1. What this builds

Five files under `src/benchmarks/recovery/` that take a conforming dataset and return the within-session reliability ceiling for a functional measure, together with the controls that make that number interpretable.

The ceiling answers one question: **can this measure distinguish a preparation from a repeat of itself?** Without it, a failed cross-session result is uninterpretable, because a noisy measure and an unstable preparation produce the same number.

This is the offer made to external labs. It must run on their data with no new experiments, and it must produce the same number in their hands as in ours, which is why the whole thing is config-driven and provenance-stamped.

### What already exists

`src/benchmarks/recovery/` has the data layer:

- `probe.py` — inventory and header extraction without reading raw voltages
- `step0.py` — stimulation table parsing, candidate filtering, blanking proposals

`src/data/ephys/` has loaders for FinalSpark, Maxwell, 3Brain and pharmacological shock, all tested.

**Gate 1 has already been run.** On 2026-10-20 it executed on the fs369 adjacent pair, 2025-02-20 and 2025-02-21, and failed at 0.128 against the 0.70 threshold. The within-session ceiling came back at 0.779 and the positive control at 0.878, which is what made the failure interpretable rather than ambiguous.

That means **the core of `evaluate.py` already exists**. Step 1b of `3_recovery_profiling_gate1.md` specifies exactly the split-half with Spearman-Brown correction specced below, and it ran. The code lives in `scripts/gate1_run.py` with outputs in `output/gate1/`.

So this is not a greenfield build. It is extraction and generalization: lifting a working one-off analysis into the `contract / adapters / evaluate / controls / run` shape so that it runs on a dataset we did not prepare. `src/benchmarks/lifespan/` has the full pattern and is the template throughout.

### What this is not

Not cross-session reliability. Not a release criterion. Not a claim about any tissue. This measures the measure.

---

## 2. The data contract

Two tables, mirroring the `(animals, series)` shape in `lifespan/contract.py`.

### `units` — one row per recording unit

| column | type | meaning |
|---|---|---|
| `unit_id` | str | electrode, channel or ROI identifier, unique within recording |
| `recording_id` | str | 12-char deterministic id from `probe._recording_id` |
| `session_id` | str | one continuous acquisition |
| `preparation_id` | str | the physical culture, well or construct |
| `x`, `y` | float | position in microns, nullable |

### `responses` — one row per (unit, trial)

| column | type | meaning |
|---|---|---|
| `unit_id` | str | foreign key into `units` |
| `trial_index` | int | 0-based, ordered within session |
| `stim_time_ns` | int64 | stimulus onset, nanoseconds since epoch |
| `condition` | str | stimulus parameter set, exact-match key |
| one column per measure | float | the quantity under test |

Measures are named in config, not hardcoded. A dataset may carry several and the suite evaluates each independently.

### The six requirements, as code

`validate()` raises on any of these. They restate `minimum-dataset.md`, and this function is the reason that document stops being prose.

1. **Single session.** `responses` joined to `units` yields exactly one `session_id`. Multi-session input is a caller error, not something to average over.
2. **Trial count.** At least 10 trials per `(unit_id, condition)` pair. Below that, a split-half correlation is noise.
3. **Unit count.** At least 15 units present in `units` and appearing in `responses`. Eight is workable with a warning; below eight, raise. The threshold is 15 rather than 20 because the fs369 reference run had 19 active electrodes, and the reference dataset must not fail its own contract. Note that `gate1_status.md` says 24 of 32 active while `3_recovery_profiling_gate1.md` says 19 above 0.1 Hz. The two documents disagree, presumably on threshold, and whichever goes public needs one number.
4. **Stimulus timestamps.** `stim_time_ns` present, non-null, strictly increasing within `(unit_id, condition)`.
5. **Trial-level data.** One row per trial. If `responses` has exactly one row per `(unit_id, condition)`, the caller has supplied summary data and the check is impossible. Raise with that diagnosis explicitly.
6. **Post-stimulus window.** Each recording carries `post_stim_window_s` in `units` metadata or config. Must be at least 2.0 seconds. This is the requirement most likely to disqualify an external dataset, so the error message must say what was found and what was needed.

### Forbidden columns

`validate()` takes `extra_forbidden: set[str]` as in lifespan. The default forbidden set for measure columns is `{condition, session_id, recording_id, stim_time_ns, trial_index}`. Passing `condition` as a measure inflates the ceiling by encoding the thing being centered out.

---

## 3. File specifications

### `recovery/contract.py` — budget 80 lines

```python
def validate(
    units: pd.DataFrame,
    responses: pd.DataFrame,
    measures: list[str],
    post_stim_window_s: float | None = None,
    extra_forbidden: set[str] | None = None,
) -> dict:
```

Raises `ContractViolation` with a message naming the requirement number, what was found, and what was needed. Returns a dict of the counts it established, which `run.py` writes into the result record.

No side effects, no file access, no config reading. Takes DataFrames, raises or returns.

### `recovery/adapters.py` — budget 220 lines

Module-level `ADAPTERS` registry mapping name to function, each taking `cfg` and returning `(units, responses, measures)`. Follows `lifespan/adapters.py` exactly.

```python
def finalspark_fs369(cfg): ...
def maxwell_hdmea(cfg): ...
def threebrain_brw(cfg): ...
def synthetic_positive(cfg): ...
def synthetic_null(cfg): ...

ADAPTERS = {...}
```

The real adapters delegate to the existing loaders in `src/data/ephys/` and to `step0.filter_candidate` for condition assignment. They do not re-implement file reading.

`synthetic_positive` generates units whose responses carry a **known** split-half reliability. Signal plus independent per-trial noise, with the variance ratio set so the expected corrected correlation equals `cfg.injected_r`. This is the positive control, and it enters through the same path as real data, which is the whole point: it exercises the contract, the centering, the splitting and the correction, not just the correlation function.

`synthetic_null` generates responses with no unit-level structure at all. Expected corrected correlation is zero.

### `recovery/evaluate.py` — budget 200 lines

```python
def center_by_condition(
    responses: pd.DataFrame,
    measures: list[str],
) -> pd.DataFrame:
```

Subtracts the condition mean from each response, per measure. Returns a copy.

Without this, variance between conditions inflates apparent reliability of the measure itself. In the fs369 pilot the uncentered ceiling was 0.779 and the centered ceiling 0.76, and only the centered figure is comparable to a centered cross-session result. The suite reports both so the gap is visible.

```python
def split_half_ceiling(
    responses: pd.DataFrame,
    measure: str,
    n_splits: int = 100,
    center: bool = True,
    shuffle_labels: bool = False,
    seed: int = 42,
) -> dict:
```

For each of `n_splits` random partitions of trials into two halves within each `(unit_id, condition)`:

1. Compute the per-unit mean of `measure` in half A and half B
2. Pearson correlate the two vectors across units
3. Spearman-Brown correct: `r_sb = 2r / (1 + r)`

Returns median and interquartile range of `r_sb` across splits, plus `r_raw`, `n_units`, `n_trials_median`, and the parameters used.

`shuffle_labels=True` permutes the `unit_id` assignment within each condition before splitting, then runs the identical code path. This is the null, and it is a flag rather than a separate function for the same reason `run_evaluation_pipeline` takes `shuffle_y`: the null must exercise every line the real analysis does.

```python
def independence_census(
    units: pd.DataFrame,
    responses: pd.DataFrame,
) -> dict:
```

Returns counts at every level, in order: preparations, sessions, units, conditions, trials. Not a statistic. The output goes into the result record so that any reported n is accompanied by what else could have been counted instead. Most reported sample sizes in this literature count trials, which inflates confidence by an order of magnitude or more.

### `recovery/controls.py` — budget 60 lines

```python
def check_controls(results: dict) -> list[str]:
```

Same signature and contract as `lifespan/controls.py`. Returns a list of failure strings, empty if everything passed. Checks:

- Positive control recovered within `tolerance` of `injected_r`
- Null median corrected correlation below `null_max`
- Null 95th percentile below the observed ceiling
- Centered and uncentered ceilings both present in the record
- `n_splits` actually achieved matches the requested count

### `recovery/run.py` — budget 280 lines

Argparse plus YAML config, following `run_lifespan_suite.py`. Responsibilities in order:

1. Load config, resolve the adapter by name from `ADAPTERS`
2. Call the adapter, get `(units, responses, measures)`
3. Call `contract.validate`, abort on violation with the requirement number
4. Run `independence_census`
5. For each measure: ceiling centered, ceiling uncentered, null
6. Run the positive control and null adapters through the identical path
7. Call `check_controls`
8. Evaluate gates
9. Write `output/recovery_ceiling/result.json` with `get_git_commit()` stamped in

Reuse `get_git_commit()` from the lifespan suite rather than reimplementing. Provenance is the difference between a number and a citable number: a result that names a commit means two labs reporting 0.8 mean the same thing.

---

## 4. Gates

Explicit named functions returning a verdict, following `evaluate_gate1/2/3` in the lifespan suite. Run the test, take the result, stop or proceed.

```python
def evaluate_gate_ceiling(r_sb_centered: float) -> str
```
PASS if `r_sb_centered >= 0.70`. The measure can distinguish a sample from a repeat of itself, and a cross-session result is therefore interpretable.
FAIL means stop. A cross-session number computed on this measure says nothing about tissue.

```python
def evaluate_gate_positive(recovered: float, injected: float, tol: float) -> str
```
PASS if `abs(recovered - injected) <= tol`, default tolerance 0.05. The pipeline recovers a signal known to be present.
FAIL means there is an analysis bug and every other number in the run is void.

```python
def evaluate_gate_null(null_p95: float, observed: float) -> str
```
PASS if `null_p95 < observed` and `null_median <= 0.1`. The result is not structural artifact.
FAIL means the observed ceiling is reachable by shuffled data and carries no information.

Gate order is positive control, then null, then ceiling. A failed positive control voids the others, so it runs first even though the ceiling is what the caller wants.

The six contract requirements are not gates. They are preconditions, enforced by `validate()`, because a dataset that fails them cannot produce an interpretable number regardless of measurement quality. The fs369 pilot is the worked example: stimulation on 7 of 21 days with no two adjacent meant the question could not have been answered at any measurement quality.

---

## 5. Config schema

`configs/recovery_ceiling.yaml`, following the conventions in `recovery_gate1.yaml`.

```yaml
adapter: "finalspark_fs369"
input_path: "data/ephys/finalspark/fs369data/fs369_package.hdf5"
forbidden_input_names: ["fs369_raw.hdf5"]
output_dir: "output/recovery_ceiling"

# fs369 measures, from the exponential fit in 3_recovery_profiling_gate1.md.
# The conditioning-burst measures (paired_pulse_ratio, fraction_recovered,
# suppression_depth, auc) belong to the newer protocol and are not
# computable on this dataset.
measures:
  - "tau"
  - "amplitude"
  - "residual_9s"
  - "time_to_1p5x_baseline"

contract:
  min_trials_per_condition: 10
  min_units: 15
  min_post_stim_window_s: 2.0

ceiling:
  split_mode: "odd_even"   # or "random"
  n_splits: 100            # ignored when split_mode is odd_even
  seed: 42

controls:
  injected_r: 0.85
  tolerance: 0.05
  null_max: 0.10

gates:
  ceiling_threshold: 0.70
```

---

## 6. Tests

Mirror `tests/lifespan/`, under `tests/benchmarks/recovery/` alongside the existing `test_probe.py` and `test_step0.py`.

**`test_contract.py`** — one test per requirement, each constructing a minimal frame that violates exactly that requirement and asserting the error message names it. Plus one frame that satisfies all six.

**`test_evaluate.py`**
- `center_by_condition` leaves within-condition structure intact and zeroes condition means
- `split_half_ceiling` on data with no unit structure returns approximately zero
- `split_half_ceiling` with `shuffle_labels=True` returns approximately zero on data that scores high unshuffled
- Spearman-Brown correction is applied: raw and corrected differ in the expected direction
- `independence_census` counts match a hand-built frame

**`test_controls.py`** — `check_controls` returns empty on a clean result dict and names the specific failure on each corrupted variant.

**The test that matters most:** the synthetic positive control recovers its injected reliability within tolerance, end to end through the adapter, the contract and the evaluation. If that passes, the pipeline is sound. If it fails, nothing else in the suite means anything.

---

## 7. Out of scope

- Cross-session reliability. Separate design, and it depends on this one passing.
- Fatigue slope. Needs a separate acquisition, not a separate analysis.
- Any claim about a tissue lot. This qualifies the measurement.
- Artifact blanking. Already handled by `step0.py`, which proposes windows rather than choosing them.
- Measure extraction. The adapters receive measures already computed. Computing paired-pulse ratio from raw voltage is upstream work and belongs in `src/data/ephys/`.

---

## 8. Open questions

1. ~~**Split unit.**~~ Resolved. `3_recovery_profiling_gate1.md` specifies odds and evens, and that is what produced the 0.779 figure. With 10 trials per condition there are only 126 distinct half-splits, so random splitting nearly enumerates the space anyway. Default to `odd_even` for reproducibility against the existing result, keep `random` available, and run both once on fs369 to confirm they agree.

1b. **The day 1 preconditioning confound, which outranks everything else here.** Day 1 had three hours of maximum-amplitude stimulation ending 7.4 hours before its sweep. Day 2 had nothing comparable. "The tissue changed overnight" and "day 1 was preconditioned" both explain the 0.128 and imply opposite things. Baseline firing also moved about 1.5x between the two days, consistent with a culture 21 days from death. Any public writeup has to state this, because it means the failure cannot be attributed to tissue instability. It is also an argument for the stimulation-history fields in the contract: a dataset that does not record what happened to the preparation before the session cannot support this comparison.

2. **Measures with censored values.** The fs369 slow-band `t_return` had 21% undefined values. Dropping them biases toward units that recovered. Options are pairwise deletion, imputation at the window edge, or reporting the censoring fraction and refusing to gate on measures above some threshold. Current preference is the third.

3. **Units versus preparations as the correlation axis.** The ceiling correlates across units within one preparation. If units within a preparation are strongly dependent, the effective n is smaller than the unit count. The independence census reports this but does not currently correct for it.

4. **External adapters.** A lab contributing data will not match the FinalSpark or Maxwell layout. Decide whether to require a CSV in the contract shape, which is simple and puts the work on them, or to accept arbitrary HDF5 with a column mapping in config, which is more work here and easier for them. The second is better for adoption.

---

## 9. Budget

| file | lines | notes |
|---|---|---|
| `contract.py` | 80 | lifespan's is 49; six requirements versus one |
| `adapters.py` | 220 | lifespan's is 184; three real plus two synthetic |
| `evaluate.py` | 200 | lifespan's is 176 |
| `controls.py` | 60 | lifespan's is 31 |
| `run.py` | 280 | lifespan's suite is 316 |
| tests | 300 | three files |
| **total** | **1140** | |

Much of `evaluate.py` and the positive control is transcription from `scripts/gate1_run.py` rather than new work, so the real cost is below this. Budget the difference against the adapters, which are the genuinely new part and the part that decides whether an external lab can use any of it.

Build order: contract, then synthetic adapters, then evaluate, then controls, then the real adapters, then run. The synthetic adapters come before evaluate so that evaluate has something with a known answer to be written against.

**First task, before any of the above:** confirm whether `scripts/gate1_run.py` is committed. This repomix packed only `src/**/*.py`, `tests/**/*.py`, `Makefile` and `configs/**/*.yaml`, so `scripts/` was filtered out of the export and its status in the repo is unknown from here. If the Gate 1 result is reproducible only from an uncommitted local script, that is the single biggest gap in the whole program, because the number quoted to external labs would not be reproducible by anyone including us.
