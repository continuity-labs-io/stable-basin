Fix src/benchmarks/recovery/probe.py. Do not touch Stage C or later. Do not
add features beyond what is listed. Keep the headers-only discipline.

GOVERNING RULE (apply everywhere):
Never substitute a default for a value that should be read from the file.
If a required value is absent, set probe_error and return the probe with
that field as None. Silent fallbacks are the bug being fixed. Remove the
`sample_rate_hz = 20000.0` fallbacks in probe_maxwell and probe_threebrain
and replace them with probe_error = "sample_rate not found in <keys tried>".

TASK 1 — duration threshold is wrong
In probe(), the usability check uses census_skip_s alone. A usable window
needs skip plus window length. Change the signature to accept
census_window_len_s and set probe_error when
duration_s <= census_skip_s + census_window_len_s.
Pass census_window_len_s through from the config in run_census_probe.
Expected effect: the 3Brain file (89.9 s) must now carry a probe_error.

TASK 2 — MaxWell sample rate
Before falling back, also try these in order and record which one hit:
  /sig attrs: "sampling_rate", "sampling", "samplingRate", "fs"
  /settings/sampling, /settings/sampling_rate, /sampling_rate
  /recordings/rec0000/well000/settings/sampling  (MaxWell v2 layout)
  /data_store/data0000/settings/sampling          (MaxWell v2 layout)
  /version, /recordings  -> log the full top-level key tree on failure
If none hit, probe_error per the governing rule. Also add a new field
`maxwell_layout` recording whether the file is v1 (/sig at root) or v2
(/recordings/... or /data_store/...).

TASK 3 — FinalSpark duration is an experiment span, not a recording
Current code reports 492083 s for fs437, which is 5.7 days of whole-life
export treated as one recording. Replace with session segmentation:
  - Read ONLY the time_of_event and electrode columns. Use
    pd.read_hdf(path, key, columns=[...]) or h5py slicing. Do not load the
    whole table.
  - Sort times. Split into sessions wherever the gap between consecutive
    events exceeds `finalspark_session_gap_s` (add to config, default 600).
  - Emit ONE RecordingProbe PER SESSION, not one per file. Give each a
    recording_id of sha256(f"{source}:{path}:{session_index}")[:12] and add
    a `session_index` field (None for single-session sources).
  - duration_s per session = last minus first event time in that session.
This changes probe() and run_census_probe() to return a list of probes per
entry rather than one. Update accordingly.

TASK 4 — FinalSpark has_xy is an unverified false negative
The current XY search only walks top-level objects one level deep and never
inspects the metadata key or the events table columns. Add, in order:
  - the events table's own column names (x, y, electrode_x, electrode_y,
    pos_x, pos_y)
  - every key matching *metadata*: dump its full column list or dtype.names
    into assumptions as `metadata_fields`, then look for x/y there
  - recurse the full HDF5 tree with h5py Group.visit(), not a one-level loop
If still not found, set xy_provenance = "absent_verified" and put the
complete visited key list in assumptions so the negative is auditable.

TASK 5 — 3Brain coordinates are grid indices, not microns
Split has_xy into two fields: `has_xy` (any coordinates at all) and
`xy_units` with values "um" | "grid_index" | "absent". 3Brain without
3BRecInfo/3BMeaChip/Pitch is "grid_index", not the same thing as MaxWell's
mapping dataset. Update xy_provenance accordingly.

TASK 6 — FinalSpark has more tables than the probe reports
The fs437 file contains fs437_wholelife_stimulations,
fs437_wholelife_incubator_CO2, _O2, _temperature, and _door_opening.
Add these fields and populate them without loading full tables:
  has_stim_table, n_stim_events, stim_time_min, stim_time_max
  has_env_channels (bool), env_channel_names (list)
These are stimulation events and environmental covariates. Report counts and
time ranges only. Do not analyze them.

TASK 7 — root config is fragile
threebrain root is "data/ephys", the parent of the finalspark and
pharmacological_shock roots, and maxwell/organoid matching excludes files by
testing for the substring "finalspark" in the path. Replace with explicit
per-source glob patterns in the config, e.g.
  finalspark:      ["**/*_package.hdf5"]
  maxwell:         ["**/*.raw.h5"]
  organoid_pharma: ["**/*.raw.h5"]
  threebrain:      ["**/*.brw"]
and resolve each source only under its own root. Raise at startup if two
roots resolve to the same file.

TASK 8 — tests
Add tests/benchmarks/recovery/test_probe.py. Build synthetic HDF5 fixtures
with h5py in a tmp_path (a few channels, a few hundred frames — tiny). Cover:
  - MaxWell v1 with sampling rate present -> correct rate and duration
  - MaxWell with no sampling rate anywhere -> probe_error set, rate None
  - a recording shorter than skip+window -> probe_error set
  - 3Brain with Pitch -> xy_units "um"; without Pitch -> "grid_index"
  - FinalSpark events spanning two sessions separated by a 2-hour gap
    -> two probes emitted with correct per-session durations
  - inventory() with two roots that would double-match -> raises
Change logger.info to logger.warning on probe failure in probe().
Remove the dead rec_id/group_token computation at the top of probe() that is
discarded unless an exception fires.

ACCEPTANCE:
Re-run `make census` (probe stage only). The new probe_table.csv must show:
  - a probe_error on the 3Brain row
  - either a real sample_rate_source for the five organoid files or a
    probe_error on each
  - more than one FinalSpark row if fs437 contains multiple sessions
  - xy_units populated for every row
Print the updated probe_assumptions.json and the new probe_table.csv.
Do not proceed past the probe stage.
