ECHO Worm Gait: Data Infra

Context: We are building infra for a new benchmark (see project docs:
"ECHO / Worm Gait — Locomotor Limit-Cycle Decline") that will eventually
compare a Gaussian EBM vs. a learned multimodal EBM on C. elegans gait
data. This prompt covers ONLY the data layer. Do not touch
src/echo/architecture/, src/echo/harness/, src/echo/primitives/,
configs/, or src/echo/benchmarks/ — those come in later prompts.

Before writing anything, read these existing files to match repo
conventions exactly:
  - src/data/ephys/pharma_shock_dataset.py   (real-dataset pattern:
    PROJECT_ROOT-relative path resolution, opening file handles inside
    __getitem__ for multiprocessing safety, raising FileNotFoundError
    with a clear message)
  - src/echo/data/toy/muller_brown.py        (synthetic/toy-dataset
    pattern: JAX simulation via jax.lax.scan + jax.vmap, wrapped in a
    torch.utils.data.Dataset that converts to torch tensors once at
    __init__ time)

Build the following:

1. src/data/behavior/__init__.py
   New package, empty (or re-export CElegansGaitDataset /
   split_by_age_cohort), matching the style of
   src/data/sim2real/__init__.py.

2. src/data/behavior/celegans_gait_dataset.py

   a) CElegansGaitDataset(Dataset)
      - Real-data loader for the Open Worm Movement Database
        (https://zenodo.org/communities/open-worm-movement-database).
        Constructor: (base_path=None, strain="N2", seq_len=1500).
        Resolve base_path relative to PROJECT_ROOT exactly like
        PharmacologicalShockDataset, defaulting to
        data/behavior/celegans_gait/.
      - Expect a hand-curated manifest at
        data/behavior/celegans_gait/manifest.csv with columns:
        filename, strain, age_days, plate_id
        (Zenodo doesn't expose age as a uniformly queryable field
        across records, so age labeling is a manual curation step —
        document this in a docstring, don't try to scrape it.)
      - __len__ = number of manifest rows matching `strain`.
      - __getitem__(idx) -> {"x_raw": eigenworm_seq, "age_days": age,
        "mask": mask}, eigenworm_seq shape [seq_len, 6] float32,
        truncated (not padded) to seq_len, mask all-ones for now.
      - Implement a private _load_eigenworm_projection(filepath) that
        tries, in order: h5py (for HDF5/v7.3 .mat), scipy.io.loadmat
        (older .mat), json (WCON). Before finalizing which field path
        to read (things like worm.posture.eigenProjection), write a
        small throwaway inspection script, download ONE real sample
        file from the Zenodo community above, run the script, and
        print the full nested key/field structure. Use what you
        actually find — do not guess the schema. Leave the inspection
        script in scripts/inspect_worm_feature_file.py so we don't
        have to redo this later.
      - Raise FileNotFoundError with a clear message (base_path or
        manifest.csv missing) — same graceful-skip contract as
        PharmacologicalShockDataset.

   b) split_by_age_cohort(dataset, young_range, old_range) ->
      (young_subset, old_subset)
      - Dataset-agnostic: works on anything exposing per-item
        "age_days" (both CElegansGaitDataset and the synthetic
        dataset below). Return torch.utils.data.Subset pairs built
        from age_days falling in young_range=(lo,hi) /
        old_range=(lo,hi) inclusive. Raise a clear error if either
        range matches zero items.

3. src/echo/data/toy/synthetic_worm_gait.py
   Same role MullerBrownDataset plays for toy physics: lets us unit-
   test and dry-run the whole pipeline with zero dependency on the
   real database.

   a) generate_gait_trajectory(key, n_steps, dt, age: float) -> jax
      array [n_steps, 6]
      - age is a float in [0, 1] (0 = young, 1 = old), NOT meant to
        be a calibrated model of real worm biomechanics — just a
        synthetic proxy with the two qualitative properties the aging
        literature reports: reduced movement amplitude and more
        irregular stroke timing.
      - Modes 0,1 (the dominant limit cycle): amplitude(age) * cos /
        sin(phase), where amplitude(age) = amplitude_young * (1 - age
        * decline_frac) and phase evolves via
        phase[t+1] = phase[t] + 2*pi*base_freq*dt + phase_noise_std(age) * dW,
        phase_noise_std(age) = phase_noise_std_young * (1 + age *
        noise_growth_frac).
      - Modes 2-5: small-amplitude harmonics of phase (e.g.
        higher_mode_scale * sin(k*phase + offset_k)) plus i.i.d.
        Gaussian noise, so the tensor is a plausible 6D signal
        dominated by modes 0-1, matching the real eigenworm data's
        expected structure.
      - Use jax.lax.scan for the recurrence and jax.vmap to batch
        across a dataset of trajectories, exactly like
        generate_trajectory in muller_brown.py.

   b) SyntheticWormGaitDataset(Dataset)
      Constructor: (size, seq_len, dt=0.05, seed=42,
      young_ages=(1,3), old_ages=(9,13)). Assign half the dataset an
      age_days uniformly sampled from young_ages, half from
      old_ages, map each to the [0,1] `age` float linearly, generate
      all trajectories at init time (vmap, like MullerBrownDataset).
      __getitem__ returns the SAME dict shape as
      CElegansGaitDataset ({"x_raw", "age_days", "mask"}) so it's a
      drop-in substitute anywhere the real dataset is expected.

4. data/behavior/README.md
   Short manual doc: link to the Zenodo community, note to filter by
   strain N2, download feature/tracking files for a spread of adult
   ages into data/behavior/celegans_gait/, and add one manifest.csv
   row per file with the age read off the record's metadata page.

5. Tests (pytest, wherever this repo's existing tests live — follow
   its current layout/conventions):
   - SyntheticWormGaitDataset: shape checks; a monotonicity check
     that the old cohort's per-trajectory phase-increment variance is
     measurably higher than the young cohort's (sanity check that the
     age-conditioning in generate_gait_trajectory actually works).
   - split_by_age_cohort: on a small synthetic dataset with known
     ages, assert the returned subsets contain exactly the expected
     indices, and that a range matching nothing raises clearly.
   - CElegansGaitDataset: wrap construction in
     pytest.mark.skipif / catch FileNotFoundError so CI doesn't
     require the real database to be present.

Non-goals for this prompt: no changes to the physics/harness code, no
wiring into MarkovBlanketObserver, no training loop. Stop once the
tests above pass.
