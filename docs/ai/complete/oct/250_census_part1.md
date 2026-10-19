# Context Files to Feed into Antigravity IDE

Please add the following files to your active context before executing the prompt:
1. `docs/ai/design/backlog/4_census_implementation.md` (or your local path to the Stage 0 Tier 1 Census Spec)
2. `src/data/ephys/finalspark_dataset.py`
3. `src/data/ephys/maxwell_dataset.py`
4. `src/data/ephys/hdmea_dataset.py`
5. `src/data/ephys/pharma_shock_dataset.py`
6. `src/benchmarks/lifespan/contract.py`

---

# Prompt to Execute

Implement Commit 1 (Stage A Inventory and Stage B Header Probe) of the Stage 0 Tier 1 Recovery Census specification.

## Architectural & Style Constraints
- Keep all logging, print statements, and variable names subdued, calm, precise, and practical (e.g., `logger.info("Probed %d files across %d sources.", n_files, n_sources)`). Avoid dramatic phrasing, all-caps banners, or emojis.
- Do not subclass or import the PyTorch `Dataset` classes in `src/data/ephys/`. Reuse only their file-format conventions (HDF5/BRW paths and key names).
- Stage A and Stage B must never read raw voltage arrays into memory; only inspect filesystem metadata, HDF5 dataset shapes/attributes, and lightweight coordinate/event metadata.
- A probe function must never raise an unhandled exception on a malformed or incomplete file. Catch all exceptions and record `repr(e)` in `RecordingProbe.probe_error`.

## Deliverables

### 1. `configs/recovery_census.yaml`
Create the configuration file with the following structure and sensible defaults matching the repository layout:
```yaml
roots:
  finalspark: "data/ephys/finalspark"
  maxwell: "data/ephys/maxwell"
  threebrain: "data/ephys"
  organoid_pharma: "data/ephys/pharmacological_shock"

output_dir: "output/recovery_census"

census_skip_s: 60.0
census_window_len_s: 40.0
census_window_fractions: [0.15, 0.5, 0.85]
max_census_channels: 256

min_spikes_active: 5
min_active_electrodes: 20

burst_bin_ms: 10.0
burst_sigma_ms: 20.0
burst_k: 4.0
burst_min_ms: 50.0
burst_min_electrodes: 10

gate_min_bursts: 50
gate_min_recordings: 20
```

### 2. `src/benchmarks/recovery/__init__.py`
Create an empty package initializer.

### 3. `src/benchmarks/recovery/probe.py`
Implement Stage A (`inventory`) and Stage B (`probe` and per-source probers):

- **Dataclasses**:
  - `FileEntry`:
    - `path: str`
    - `source: str` (`"finalspark" | "maxwell" | "threebrain" | "organoid_pharma"`)
    - `bytes: int`
    - `mtime: float`
  - `RecordingProbe`:
    - `recording_id: str` (deterministic 12-char hex SHA-256 of `f"{source}:{path}"`)
    - `source: str`
    - `path: str`
    - `group_token: str | None` (inferred from parent directory name or filename prefix, e.g., `"fs437"` or `"Drug_2953"`)
    - `recording_timestamp: str | None` (extracted from header if available, else `None`)
    - `n_channels: int | None`
    - `n_rows: int | None`
    - `n_cols: int | None`
    - `duration_s: float | None`
    - `sample_rate_hz: float | None`
    - `has_xy: bool`
    - `xy_provenance: str` (`"mapping_dataset" | "derived_from_grid" | "absent"`)
    - `has_spike_table: bool`
    - `probe_error: str | None`

- **Stage A (`inventory(roots: dict[str, str]) -> list[FileEntry]`)**:
  - Walk each root directory if it exists (if a root does not exist, log a calm info message and continue).
  - Match source-specific file patterns so auxiliary files are not double-counted:
    - `finalspark`: match files ending with `_package.hdf5` (do not separately inventory `_raw.hdf5` or `.parquet` index files).
    - `maxwell` and `organoid_pharma`: match `.raw.h5` (or `.h5` if not inside a finalspark folder).
    - `threebrain`: match `.brw`.
  - Return a deterministically sorted list of `FileEntry` objects.

- **Stage B (`probe(entry: FileEntry, census_skip_s: float = 60.0) -> tuple[RecordingProbe, dict]`)**:
  - Dispatch to `probe_maxwell` (for both `maxwell` and `organoid_pharma`), `probe_threebrain`, or `probe_finalspark`.
  - Return both the `RecordingProbe` and a dictionary of confirmed format assumptions (e.g., which HDF5 key held the signal, which key/fallback provided the sample rate, what fields were found in `/mapping`, or what grid dimensions/pitch were found).
  - If `duration_s is not None and duration_s <= census_skip_s`, set `probe_error = f"duration_s ({duration_s:.1f}s) <= census_skip_s ({census_skip_s:.1f}s)"`.
  - Wrap the dispatch in a `try...except Exception as e` block that returns a `RecordingProbe` with `has_xy=False`, `xy_provenance="absent"`, `has_spike_table=False`, and `probe_error=repr(e)`.

- **Per-Format Probers**:
  - `probe_maxwell(entry: FileEntry)`:
    - Open with `h5py.File(entry.path, "r")`.
    - Locate the voltage dataset using the first present key in `["/sig", "sig", "/routing/lsb", "/mapping/sig"]`. Extract `n_channels, n_frames = sig.shape` (without reading the array).
    - Check for sampling rate in HDF5 attributes or datasets (e.g., `sig.attrs.get("sampling_rate")`, `/settings/sampling`, `/sampling_rate`), defaulting to `20000.0` Hz and recording the exact provenance in the assumptions dict.
    - Check `/mapping` (or `mapping`): verify whether `"x"` and `"y"` exist either in `mapping.dtype.names` (structured array) or as sub-datasets. Set `has_xy` accordingly and `xy_provenance = "mapping_dataset" if has_xy else "absent"`.
  - `probe_threebrain(entry: FileEntry)`:
    - Open with `h5py.File(entry.path, "r")`.
    - Inspect `3BData/Raw` shape to get `n_elements = f["3BData/Raw"].shape[0]`.
    - Read `n_rows` and `n_cols` from `3BRecInfo/3BMeaChip/NRows` and `3BRecInfo/3BMeaChip/NCols` if present, or fall back to `(64, 64)` if `n_elements % 4096 == 0`, recording the exact source in the assumptions dict.
    - Read `sample_rate_hz` from `3BRecInfo/3BRecVars/SamplingRate` if present, or fall back to `20000.0`, recording provenance in assumptions.
    - Check for pitch metadata in `3BRecInfo/3BMeaChip` (or record `"grid_indices_only"` in assumptions). Set `has_xy = (n_channels is not None and n_channels > 0 and n_rows * n_cols == n_channels)` and `xy_provenance = "derived_from_grid" if has_xy else "absent"`.
  - `probe_finalspark(entry: FileEntry)`:
    - Inspect keys in `entry.path`. Read the events table (`fs437_wholelife_events` or matching `*_events` key) using `pd.read_hdf`.
    - Compute `n_channels = int(events["electrode"].nunique())`.
    - Parse `time_of_event` as UTC timestamps and compute `duration_s` as `(max_t - min_t).total_seconds()`, and `recording_timestamp` as `min_t.isoformat()`.
    - Check if any key or table in the HDF5 file contains electrode `(x, y)` coordinates (`"x"` and `"y"` columns/datasets). Set `has_xy` accordingly, `xy_provenance = "mapping_dataset" if has_xy else "absent"`, and `has_spike_table = True`.

- **CLI Entry Point (`if __name__ == "__main__":`)**:
  - Accept `--config` (default `"configs/recovery_census.yaml"`).
  - Run `inventory()` and `probe()` across all discovered files.
  - Write `output/recovery_census/probe_table.csv` (one row per file) and `output/recovery_census/probe_assumptions.json` (summary of counts per source, total bytes, and confirmed format assumptions).
  - Log a concise summary per source: file count, total bytes, count with `has_xy=True`, and count with `probe_error is not None`.

### 4. `tests/test_probe.py`
Write fast unit tests using `pytest` and `tmp_path`:
- `test_inventory_counts_and_filters_extensions`: verify that `fs437_package.hdf5`, `.raw.h5`, and `.brw` are found while `fs437_raw.hdf5` and unrelated files are ignored.
- `test_probe_maxwell_with_and_without_xy`: create a tiny 2
