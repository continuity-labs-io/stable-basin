# Stage 0 Tier 1 — Census Implementation Spec

How to run the tier 1 gate from the retrospective HD-MEA design doc, against the code that exists today.

## Summary 

The whole design is built so you can quit early. The first two stages never open a recording — they just list the files and read their labels. That takes minutes and answers the question most likely to end this: whether the data records where each electrode sits on the chip. Without that, half the planned analysis is impossible. FinalSpark is the most likely source to fail it, and you’ll know by lunchtime rather than after a week of building.

Don’t reuse the data code you already have. It was written to feed models, not to count things, so it throws away exactly what the census needs. Worth keeping is what it knows about each file format — that part saves real time.

The rough counting tool stays walled off from the real analysis. It’s right for “is anything happening here” and wrong for everything downstream. Two of your past negative results turned out to be bugs rather than biology, and mixing these two jobs is how that happens a third time.

One caution on the result. Twenty recordings from one chip isn’t twenty independent recordings, so the verdict reports the breakdown by source, not just a pass or fail.

First thing to build: feed the counter pure noise and confirm it finds nothing. If it finds structure in noise, it will pass the gate on worthless data and cost you two weeks.


## The governing principle

**The census is a probe, not a load.** It answers structural questions about files — how many, how long, how many channels, do electrode positions exist, is there network bursting — and it must never require the `Recording` loader to be finished. If the census depends on the loader, the cheap gate is blocked behind the expensive work, which defeats the entire purpose of having a gate.

Two consequences, both load-bearing:

- The probe reads headers and metadata. It does not read voltage except on a bounded sample window — bounded in both time and channel count — and only in the last stage.
- Census-grade spike and burst detection is deliberately crude and must be **quarantined in its own module, clearly named, and never imported by tier 2 or above.** A census detector leaking into the analysis pipeline is precisely the failure mode that produced two uninterpretable nulls already.

## What the existing code gives you

Every ephys loader in `src/data/ephys/` is a PyTorch `Dataset` that returns normalized voltage tensors for model training. **None of them produce spikes, electrode positions, or metadata.** They are the wrong shape for this job and should not be subclassed or extended.

What is reusable is the file-format knowledge inside them.

| Source | File | Known key | Reusable from |
| --- | --- | --- | --- |
| FinalSpark | `fs437_package.hdf5` + `fs437_raw.hdf5` + parquet index | `fs437_wholelife_events` (electrode, time_of_event) | `finalspark_dataset.py` |
| MaxWell | `.raw.h5` | `/sig`, `/routing/lsb`, `/mapping/sig` | `maxwell_dataset.py` |
| 3Brain | `.brw` | `3BData/Raw`, flat, 4096 channels | `hdmea_dataset.py` |
| Organoid pharmacology | `Drug_2953_{condition}.raw.h5` | `sig`, shape (channels, time) | `pharma_shock_dataset.py` |

Two asymmetries worth planning around.

**FinalSpark already has detected spikes.** The `fs437_wholelife_events` table is an event list with electrode and timestamp. No detection needed — the census just counts. This dataset will be cheap and will finish first.

**The organoid pharmacology set is MaxWell format**, so it shares a probe implementation with the MaxWell recordings. Two sources, one code path.

## Where electrode positions are expected to live

The probe's job is to verify these, not to trust them. Each is an assumption to confirm on the first file and record in the probe output.

- **MaxWell `.raw.h5`** — a `/mapping` dataset with per-channel `x` and `y` in µm. This is the standard export layout; confirm the field names on a real file.
- **3Brain `.brw`** — not stored directly. Expected to be derivable from `3BRecInfo/3BMeaChip` row/column counts plus the chip pitch, with channel index mapping to a grid position. Confirm the pitch for the specific chip before trusting any µm value.
- **FinalSpark** — electrode identity exists in the events table, but physical layout may not ship with the export. **This is the most likely dataset to fail the XY requirement**, and finding that out is a stage B result, not a stage D surprise.

Per the design doc, a dataset that cannot supply positions is out of scope rather than partially usable. That verdict is reached in stage B, before any voltage is read.

## Layout

```
src/benchmarks/recovery/
    __init__.py
    probe.py              # probe contract + one prober per format
    census_detect.py      # census-grade detection ONLY — see the header warning
    census.py             # orchestrator, writes the table
    contract.py           # validates the census table, mirrors lifespan/contract.py
configs/
    recovery_census.yaml
output/recovery_census/
    census.csv
    census.json           # sidecar: code version, params, seed, probe assumptions confirmed
    verdict.txt
tests/
    test_probe.py
    test_census_detect.py
```

Makefile, matching the house pattern:

```make
census:
	python -m src.benchmarks.recovery.census --config configs/recovery_census.yaml
```

## Stage A — Inventory

Pure filesystem. No file is opened. Runs in seconds and catches "the data is not where I think it is" before anything else.

```python
@dataclass
class FileEntry:
    path: str
    source: str          # finalspark | maxwell | threebrain | organoid_pharma
    bytes: int
    mtime: float

def inventory(roots: dict[str, str]) -> list[FileEntry]:
    entries = []
    for source, root in roots.items():
        for path in walk(root):
            if not matches_extension(path, EXT_BY_SOURCE[source]):
                continue
            entries.append(FileEntry(path, source, size_of(path), mtime_of(path)))
    return entries
```

**Output:** a count per source and total bytes. **Stop here if** any source has zero files — that is a data-acquisition problem, not an analysis problem, and no further stage will tell you anything new.

## Stage B — Probe (headers only, no voltage)

The expensive question answered cheaply. Open each file, read metadata, close.

```python
@dataclass
class RecordingProbe:
    recording_id: str          # stable hash of source + relative path
    source: str
    path: str
    group_token: str | None    # from folder or filename
    recording_timestamp: str | None  # if present in header
    n_channels: int | None
    n_rows: int | None         # grid formats (3Brain); used by Stage C 2D channel stride
    n_cols: int | None
    duration_s: float | None
    sample_rate_hz: float | None
    has_xy: bool
    xy_provenance: str         # "mapping_dataset" | "derived_from_grid" | "absent"
    has_spike_table: bool      # FinalSpark only, today
    probe_error: str | None    # populated instead of raising

def probe(entry: FileEntry) -> RecordingProbe:
    try:
        return PROBERS[entry.source](entry)
    except Exception as e:
        return RecordingProbe(..., probe_error=repr(e))
```

**A probe never raises.** It records the failure in the row. A single malformed file must not abort a census over hundreds, and the count of probe failures is itself a result.

Per-format work:

```python
def probe_maxwell(entry) -> RecordingProbe:
    with h5py.File(entry.path, "r") as f:
        sig = first_present(f, ["/sig", "/routing/lsb", "/mapping/sig"])
        n_channels, n_frames = sig.shape
        fs = read_sample_rate(f)            # confirm the key on a real file
        xy = f.get("/mapping")
        has_xy = xy is not None and {"x", "y"} <= set(field_names(xy))
    return RecordingProbe(..., duration_s=n_frames / fs,
                          has_xy=has_xy,
                          xy_provenance="mapping_dataset" if has_xy else "absent")

def probe_threebrain(entry) -> RecordingProbe:
    with h5py.File(entry.path, "r") as f:
        n_elements = f["3BData/Raw"].shape[0]
        n_channels = read_int(f, "3BRecInfo/3BMeaChip/NRows") * read_int(f, ".../NCols")
        fs = read_sample_rate(f)
        has_xy = n_channels > 0 and chip_pitch_known(f)
    return RecordingProbe(..., duration_s=(n_elements / n_channels) / fs,
                          has_xy=has_xy, xy_provenance="derived_from_grid")

def probe_finalspark(entry) -> RecordingProbe:
    events = read_hdf_key(entry.path, "fs437_wholelife_events")
    layout = try_read_electrode_layout(entry.path)
    return RecordingProbe(..., n_channels=events["electrode"].nunique(),
                          duration_s=span_seconds(events["time_of_event"]),
                          has_xy=layout is not None,
                          xy_provenance="mapping_dataset" if layout else "absent",
                          has_spike_table=True)
```

**Output:** the probe table. **Stop here if** `has_xy` is false for every file in a source — that source is out of scope and should be struck from the config before stage C, not carried forward and discovered later.

**Record the confirmed assumptions in the JSON sidecar.** Which key held the sample rate, whether `/mapping` had the expected fields, what pitch was used for the 3Brain grid. These are the things that will be wrong, and writing them down now is what makes a later bug findable in minutes instead of days.

## Stage C — Spike counts on bounded windows

First stage that reads voltage, and it reads as little as possible — in time **and** in channels.

### Time windows: three strided reads, not one contiguous block

A single contiguous `[60 s, 180 s]` read samples only the head of the recording, and extrapolating its burst rate to the full duration assumes bursting is stationary. Two sources break that directly: the organoid `Drug_2953_{condition}.raw.h5` files (wash-in lag after a condition change) and multi-hour recordings (late-recording rundown).

Stage C instead reads **three 40 s windows centred at 15%, 50% and 85% of the usable span** `[census_skip_s, duration_s]`. Same 120 s of data read, but sampled across the recording.

- **Fractions are of the usable span, not of `duration_s`.** On a 400 s recording, 15% of `duration_s` is 60 s, which puts the first window at `[40, 80]` — inside the settling period that `census_skip_s` exists to discard. Measuring from `census_skip_s` keeps every window out of it.
- **Fallback is derived from the geometry, not hardcoded.** Strided windows are used only if all three fit inside the usable span without overlapping. With `skip = 60`, `L = 40` and fractions `(0.15, 0.5, 0.85)`, that requires `0.15 × (D − 60) ≥ 20`, i.e. `D ≥ ~193 s`. Below that, the census falls back to one contiguous window `[skip, min(skip + 120, D)]` — the previous behaviour. A recording between 180 s and 193 s has ≤ 133 s of usable span, so the contiguous window already covers almost all of it.
- **Each window is processed on its own** in Stage D (smoothing, baseline, threshold) so nothing is smoothed across a gap. See Stage D.
- `windows` and `is_strided` go in the row, so every count can be traced to the exact seconds that produced it.

### Channel cap: why a time window alone is not enough

120 s of reads across every channel is still enormous. A 3Brain `.brw` has 4,096 channels at 20 kHz: 2,400,000 frames × 4,096 channels × 4 bytes = **39.3 GB of float32 per file**, followed by 4,096 sequential scipy bandpass filters. MaxWell at ~1,024 routed channels is ~9.8 GB. Either will thrash unified memory on a laptop and turn a minutes-long stage into hours.

Stage C therefore reads at most `max_census_channels = 256` channels per recording, spread evenly across the array. A network burst is a population synchronization event; the working assumption is that a burst recruiting ≥10% of the array is still detected on a uniform 256-channel subgrid, at 1/16th the memory and compute of a full 3Brain read. **That is an assumption, not a measured fidelity** — `test_burst_count_preserved_under_subsampling` below is what checks it, and the census does not cite a fidelity number until that test passes.

Three details that are easy to get wrong:

- **Stride in 2D, not on the flat index.** On the 64×64 3Brain grid, `channels[::16]` over the row-major flat index selects columns 0, 16, 32, 48 in every row — four vertical stripes, not a uniform grid. Stride rows and columns separately (`rows[::4] × cols[::4]` → 16×16 = 256). For MaxWell, routed electrodes are arbitrary, so select by stride over electrodes sorted by `(y, x)` from `/mapping`, not by channel index.
- **Subsample before converting to float.** `3BData/Raw` is a flat interleaved int16 array, so the selected channels cannot be read in isolation. Read in time chunks (e.g. 1 s ≈ 164 MB of int16 at 4,096 ch), keep the selected columns, then convert. Peak memory is one chunk, not the window.
- **Scale every electrode-count threshold by the read fraction.** `n_active_electrodes` is counted on the subsample, so `min_active_electrodes` and `burst_min_electrodes` are multiplied by `n_channels_read / n_channels` (ceil, floor of 1). Both raw and scaled thresholds go in the JSON sidecar. Sources with a native event table (FinalSpark) are not subsampled; their fraction is 1.

```python
# From configs/recovery_census.yaml:
#   census_skip_s: 60.0                       # discard settling at the head of every recording
#   census_window_len_s: 40.0
#   census_window_fractions: [0.15, 0.5, 0.85]  # centres, as fractions of [skip, duration]
#   max_census_channels: 256

def census_windows(duration_s: float, cfg) -> list[tuple[float, float]]:
    """Returns the [(start_s, end_s), ...] read windows for one recording."""
    skip, L = cfg.census_skip_s, cfg.census_window_len_s
    fracs = cfg.census_window_fractions
    usable = duration_s - skip
    starts = [skip + f * usable - L / 2 for f in fracs]
    fits = (starts[0] >= skip
            and starts[-1] + L <= duration_s
            and all(b - a >= L for a, b in zip(starts, starts[1:])))
    if fits:
        return [(s, s + L) for s in starts]
    # Fallback: one contiguous window of the same total length, or whatever exists.
    return [(skip, min(skip + len(fracs) * L, duration_s))]

def select_census_channels(probe: RecordingProbe, max_channels: int) -> list[int]:
    if probe.n_channels <= max_channels:
        return list(range(probe.n_channels))
    if probe.xy_provenance == "derived_from_grid":      # 3Brain
        return grid_stride(probe.n_rows, probe.n_cols, max_channels)  # 2D stride
    return position_stride(read_xy(probe), max_channels)  # MaxWell: sorted by (y, x)

def scaled_threshold(raw: int, channel_fraction: float) -> int:
    return max(1, ceil(raw * channel_fraction))

def census_spikes(probe: RecordingProbe, cfg) -> SpikeSample:
    windows = census_windows(probe.duration_s, cfg)
    if probe.has_spike_table:
        spikes = read_event_table_windows(probe, windows)
        n_read = probe.n_channels
        method = "native_event_table"
    else:
        chans = select_census_channels(probe, cfg.max_census_channels)
        per_window = []
        for start_s, end_s in windows:
            trace = read_voltage_window_chunked(probe, chans, start_s, end_s)  # (len(chans), frames)
            per_window.append(threshold_crossings(trace, probe.sample_rate_hz,
                                                  t0_s=start_s, channel_ids=chans))
        spikes = np.concatenate(per_window)                  # windows are disjoint and ordered
        n_read = len(chans)
        method = "mad_threshold_census_only"
    # spikes: (n_spikes, 2) array of (time_s, channel_idx), sorted by time_s.
    # time_s is absolute recording time; channel_idx is the recording's own
    # channel index, so electrode identity survives into Stage D.
    return SpikeSample(
        recording_id=probe.recording_id,
        spikes=spikes,
        windows=windows,
        is_strided=len(windows) > 1,
        window_len_s=sum(end - start for start, end in windows),   # total seconds read
        recording_duration_s=probe.duration_s,
        n_channels_read=n_read,
        channel_fraction=n_read / probe.n_channels,
        n_spikes=len(spikes),
        n_active_electrodes=count_distinct_electrodes(spikes, min_spikes=cfg.min_spikes_active),
        median_rate_hz=median_per_electrode_rate(spikes),
        detect_method=method,
    )
```

The detector, quarantined:

```python
# census_detect.py
# ---------------------------------------------------------------------------
# CENSUS GRADE ONLY. Not for tier 2 or above. This module exists to answer
# "is there activity here" and nothing else. Importing it from an analysis
# module is a bug.
# ---------------------------------------------------------------------------

def threshold_crossings(trace, fs, t0_s, channel_ids, k=5.0, refractory_ms=1.0):
    """Returns (n_spikes, 2) array of (time_s, channel_idx), sorted by time_s."""
    rows = []
    for row, ch in enumerate(channel_ids):
        x = bandpass(trace[row], 300.0, 3000.0, fs)
        sigma = median(abs(x)) / 0.6745            # MAD estimate of noise
        frames = find_negative_crossings(x, -k * sigma)
        frames = enforce_refractory(frames, refractory_ms, fs)
        rows.append(np.column_stack([t0_s + frames / fs, np.full(len(frames), ch)]))
    spikes = np.concatenate(rows)
    return spikes[np.argsort(spikes[:, 0], kind="stable")]
```

**Every recording uses the same window rule** (`census_windows` with the same config), so counts are comparable across files. Where a recording is too short for strided windows, it gets the contiguous fallback; where it is shorter than the fallback, the actual length read is recorded and rates are used instead of counts. The MAD noise estimate in `threshold_crossings` is recomputed per window, so a noise-floor change between windows does not shift the spike threshold of a later window.

## Stage D — Burst counts and extrapolation

Two passes, because the two criteria live in different shapes:

1. **Candidate windows from the 1D population rate.** Binning spikes into a population rate discards electrode identity — that is what makes it 1D. So this pass enforces only the amplitude and duration criteria: runs where the smoothed rate exceeds `baseline + k × MAD` for at least `burst_min_ms`, returned as `[t_start, t_end]` intervals in seconds.
2. **Recruitment filter on the 2-column spike array.** For each candidate interval, count `np.unique(channel_idx)` among spikes with `t_start ≤ time_s < t_end`. Keep the interval only if that count reaches the (subsample-scaled) `burst_min_electrodes`.

Applying `min_electrodes` to the 1D rate directly is not possible: a surge from one electrode firing rapidly and a surge from 40 electrodes firing once each look identical there. Without pass 2, the first case passes as a burst.

Pass 1 runs **per read window**: the rate is binned and smoothed inside each window (never across the gap between two windows), and each window gets its own `median + k × MAD` threshold. A burst is a transient above the local baseline; a per-window threshold keeps a condition-driven shift in tonic firing from turning a whole 40 s window into one long supra-threshold run.

```python
def census_bursts(sample: SpikeSample, cfg) -> BurstSample:
    t, ch = sample.spikes[:, 0], sample.spikes[:, 1]      # sorted by t
    min_elec = scaled_threshold(cfg.burst_min_electrodes, sample.channel_fraction)

    n_candidates, bursts_per_window = 0, []
    for w0, w1 in sample.windows:
        # Pass 1 — 1D rate within this window: amplitude + duration only.
        rate = population_rate(t, t_range=(w0, w1), bin_ms=cfg.burst_bin_ms)
        smoothed = gaussian_smooth(rate, sigma_ms=cfg.burst_sigma_ms, bin_ms=cfg.burst_bin_ms)
        threshold = median(smoothed) + cfg.burst_k * mad(smoothed)
        candidates = runs_above(smoothed, threshold, min_duration_ms=cfg.burst_min_ms,
                                bin_ms=cfg.burst_bin_ms, t0_s=w0)  # list of (t_start_s, t_end_s)
        n_candidates += len(candidates)

        # Pass 2 — back on (time_s, channel_idx): recruitment.
        kept = 0
        for t_start, t_end in candidates:
            lo, hi = np.searchsorted(t, [t_start, t_end], side="left")
            if len(np.unique(ch[lo:hi])) >= min_elec:
                kept += 1
        bursts_per_window.append(kept)

    observed = sum(bursts_per_window)
    rate_per_min = observed / (sample.window_len_s / 60.0)
    return BurstSample(
        n_candidate_windows=n_candidates,      # pass-1 count, before recruitment filter
        n_bursts_observed=observed,
        burst_rate_per_min=rate_per_min,
        burst_rate_per_window=[n / ((w1 - w0) / 60.0)
                               for n, (w0, w1) in zip(bursts_per_window, sample.windows)],
        n_bursts_estimated_full=rate_per_min * (sample.recording_duration_s / 60.0),
        is_extrapolated=sample.window_len_s < sample.recording_duration_s,
    )
```

`n_candidate_windows` is kept in the row so the census shows how many rate excursions the recruitment filter rejected. A large gap between the two counts on a source is worth inspecting — one possible cause is a few hot electrodes driving the population rate.

**Both the observed count and the extrapolated count are recorded, and the extrapolation flag travels with the row.** The extrapolation is the mean rate over the read windows times the full duration. With strided windows that mean is taken at early, middle and late points rather than only the head, but it is still an assumption that those three points represent the gaps between them. `burst_rate_per_window` is recorded so a recording whose rate differs sharply across windows is visible in the table rather than averaged away. Any gate decision that depends on extrapolated counts gets reported as such in the verdict.

## Stage E — Census table and verdict

One row per recording. All probe fields, all sample fields (including `windows`, `is_strided`, `n_channels_read` and `channel_fraction`), all burst fields (including `burst_rate_per_window`, serialised as a JSON list in the CSV), plus structural provenance: `recording_id`, `source`, `path`, `group_token` (from folder/filename), and `recording_timestamp` (if present in the header).

**Drop the "literature registry" columns from Tier 1.** Biological covariates (`DOI`, `licence`, `species`, `culture type`, `donor age`, `days in vitro`, `chip` and `well identifiers`, `treatment`, and `dose`) do not exist inside `.raw.h5` or `.brw` headers—they live in supplementary PDFs or ad-hoc filenames. Writing parsers or manually curating metadata for files that might fail the burst gate is pure wasted effort. Defer biological covariates to Tier 2, and only for the specific files that pass the gate.

Validate before writing, mirroring `benchmarks/lifespan/contract.py`:

```python
def validate_census(df: pd.DataFrame) -> None:
    if not df["recording_id"].is_unique:
        raise ValueError("recording_id is not unique")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"census missing required column {col}")
    if (df["window_len_s"] <= 0).any():
        raise ValueError("non-positive census window")
    if df["has_xy"].isna().any():
        raise ValueError("has_xy must be resolved true/false for every row")
```

Then the gate:

```python
def evaluate_gate(df, cfg) -> Verdict:
    # n_active_electrodes was counted on the subsample, so the threshold is scaled per row
    min_active = np.maximum(1, np.ceil(cfg.min_active_electrodes * df.channel_fraction))
    usable = df[(df.probe_error.isna()) & (df.has_xy) &
                (df.n_active_electrodes >= min_active)]
    bursting = usable[usable.n_bursts_estimated_full >= 50]

    passed = len(bursting) >= 20
    return Verdict(
        passed=passed,
        n_files=len(df),
        n_probe_failures=int(df.probe_error.notna().sum()),
        n_without_xy=int((~df.has_xy).sum()),
        n_usable=len(usable),
        n_bursting=len(bursting),
        n_bursting_extrapolated=int(bursting.is_extrapolated.sum()),
        by_source=bursting.groupby("source").size().to_dict(),
    )
```

**Gate: 20 or more recordings with electrode positions and 50 or more network bursts each.**

The per-source breakdown matters as much as the headline. Twenty bursting recordings all from one chip is not twenty independent recordings, and the verdict should make that visible rather than letting a passing number hide it.

## Tests, written before stage C

Eight synthetic fixtures, following the validation discipline in the design doc.

```python
def test_detector_finds_known_bursts():
    trace = synth_trace(n_channels=64, fs=20_000, duration_s=120,
                        burst_times=known_times, burst_width_ms=200)
    bursts = census_bursts(census_spikes_from_trace(trace))
    assert abs(bursts.n_bursts_observed - len(known_times)) <= 1

def test_detector_returns_nothing_on_matched_poisson():
    trace = synth_trace_poisson(n_channels=64, fs=20_000, duration_s=120,
                                rate_hz=matched_to_above)
    bursts = census_bursts(census_spikes_from_trace(trace))
    assert bursts.n_bursts_observed == 0

def test_grid_stride_is_spatially_uniform():
    chans = grid_stride(n_rows=64, n_cols=64, max_channels=256)
    rows, cols = divmod(np.array(chans), 64)
    assert len(chans) == 256
    assert len(np.unique(rows)) == 16 and len(np.unique(cols)) == 16  # not 4 stripes

def test_burst_count_preserved_under_subsampling():
    # Spike-train level (no voltage), so a 4,096-channel fixture stays cheap.
    spikes = synth_spike_trains_grid(n_rows=64, n_cols=64, duration_s=120,
                                     burst_times=known_times, recruit_fraction=0.10)
    full = census_bursts(spikes_on(spikes, all_channels(64, 64)))
    sub  = census_bursts(spikes_on(spikes, grid_stride(64, 64, 256)))
    assert abs(sub.n_bursts_observed - full.n_bursts_observed) <= 1

def test_single_electrode_surge_is_not_a_burst():
    # Population-rate excursion driven by one electrode: passes pass 1, must fail pass 2.
    spikes = synth_spike_trains(n_channels=64, duration_s=120, rate_hz=1.0)
    spikes = add_surge(spikes, channel_idx=7, t_start_s=90.0, duration_ms=200,
                       rate_hz=500.0)
    bursts = census_bursts(sample_from(spikes), cfg_with(burst_min_electrodes=10))
    assert bursts.n_candidate_windows >= 1      # the rate did cross threshold
    assert bursts.n_bursts_observed == 0        # but recruitment rejected it

def test_census_windows_strided_geometry():
    wins = census_windows(duration_s=3600.0, cfg)          # skip=60, L=40, (0.15, 0.5, 0.85)
    assert len(wins) == 3
    assert all(end - start == pytest.approx(40.0) for start, end in wins)
    assert wins[0][0] >= 60.0 and wins[-1][1] <= 3600.0    # inside usable span
    assert all(b[0] >= a[1] for a, b in zip(wins, wins[1:]))  # disjoint, ordered

def test_census_windows_fallback_on_short_recording():
    assert census_windows(duration_s=150.0, cfg) == [(60.0, 150.0)]  # shorter than 120 s usable
    assert census_windows(duration_s=190.0, cfg) == [(60.0, 180.0)]  # strided would not fit
    assert len(census_windows(duration_s=194.0, cfg)) == 3           # just above ~193 s

def test_strided_windows_see_late_onset_bursting():
    # Bursting starts at 1200 s. A [60, 180] read would see none; the 85% window must.
    spikes = synth_spike_trains(n_channels=64, duration_s=1800, rate_hz=1.0)
    spikes = add_bursts(spikes, burst_times=np.arange(1200.0, 1800.0, 5.0),
                        recruit_fraction=0.3)
    sample = sample_from(spikes, windows=census_windows(1800.0, cfg))
    bursts = census_bursts(sample, cfg)
    assert bursts.burst_rate_per_window[0] == 0
    assert bursts.burst_rate_per_window[-1] > 0
```

The subsampling test is what licenses the 256-channel cap. If it fails at 10% recruitment, raise `max_census_channels` in the config rather than relaxing the test.

The negative control is the one that matters. A burst detector that finds bursts in rate-matched Poisson noise will pass the tier 1 gate on nothing at all, and you will spend two weeks building a loader for data that has no structure in it.

## What this deliberately does not do

- No spike sorting. Multi-unit per electrode, per the design doc's trade-off decision.
- No `Recording` loader. That is the next stage and only if the gate passes.
- No recovery fitting, no restitution, no spatial measures. Those are tiers 2 through 4.
- No group comparison. Biological registry parsing is deferred to tier 2 and only joined to files that pass the gate.

## Expected outcome

Stage A and B run in minutes and answer the question most likely to kill the branch, which is whether electrode positions exist at all. Stage C and D are an afternoon. If the gate fails, the honest output is the one-line note the design doc already commits to, plus the census table showing exactly which constraint failed — which is also the precise shopping list for what a new dataset would need to supply.
