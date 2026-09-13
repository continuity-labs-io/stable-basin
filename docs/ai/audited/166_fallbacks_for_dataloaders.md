**Task: Implement CI/CD Deterministic Synthetic Fallbacks for DataLoaders**

**Objective:** We must completely decouple our software build velocity and
Continuous Integration (CI) pipeline from massive, external biological databases
(Open Worm DB, Sleep-EDF, MPI-LEMON). The pipeline must be able to execute
perfectly offline on a blank GitHub Actions runner without network timeouts or
massive storage requirements.

**Implementation Rules:** For every PyTorch `Dataset` class you construct in
this project (e.g., `CElegansGaitDataset`, the MNE EEG Loaders), you must
implement the following structural fallback pattern:

1. **The File Check:** In the `__init__` method, check if the raw biological
   data files or directories exist locally.
2. **The Silent Fallback:** If the files do not exist, catch the
   `FileNotFoundError` (or equivalent) and trigger a fallback flag (e.g.,
   `self.use_synthetic = True`) rather than crashing or attempting to download
   gigabytes of data.
3. **The Generator Method:** Create a static or class method called
   `generate_synthetic_data(seq_len, seed=42)`.
   - This method MUST use a strict random seed (`numpy.random.default_rng(seed)`
     or `torch.manual_seed(seed)`) to ensure the generated data is bit-for-bit
     reproducible on every single CI run.
   - **For Worm Gait (6D):** Generate a 6-dimensional limit cycle (e.g., 6
     out-of-phase sine/cosine waves with slight Gaussian noise) to
     mathematically mimic continuous eigenworm locomotion. Simulating "aging"
     can be done by increasing the noise variance.
   - **For EEG PCA:** Generate an $N$-dimensional Auto-Regressive (AR) process
     or multivariate Gaussian noise with a defined covariance structure to mimic
     PCA-reduced brain waves.
4. **Static Shapes:** The output of `__getitem__` in synthetic mode must
   perfectly match the expected tensor shapes of the real biological crops
   (e.g., `[seq_len, feature_dim]`). The JAX `@eqx.filter_jit` compiler should
   not be able to tell the difference between real and synthetic data shapes.

**Constraints:**

- Do not add massive third-party simulation libraries to generate this data. Use
  pure `numpy` or `torch` primitives to generate the math. Keep it lightweight,
  fast, and entirely deterministic.
- **Logging:** Use the standard Python `logging` module. Emit a single peaceful
  info message when falling back:
  `logger.info("Local biological data not found. Falling back to deterministic synthetic generation for CI.")`
  Avoid dramatic, capitalized, or emoji-laden print statements.
