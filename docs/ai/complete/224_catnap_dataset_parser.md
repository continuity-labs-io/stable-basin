We are implementing "Phase 1: Data Engineering" for Project Catnip. We need to create a PyTorch Dataset to parse the Calico CATNAP HDF5 data.

1. Create a new file `src/data/behavior/catnip_dataset.py`.
2. Add the required imports: `logging`, `pathlib`, `pandas as pd`, `numpy as np`, `torch`, `from torch.utils.data import Dataset`, `from torch import Tensor`, `from jaxtyping import Float, jaxtyped`, and `from beartype import beartype`.
3. Set up a module logger: `logger = logging.getLogger(__name__)`.
4. Create the `CatnipContinuousDataset(Dataset)` class.
5. In the `__init__` method, accept: `h5_path: str = "data/catnap/trace_features.h5"`, `sequence_length: int = 10`, and `cohort: str = "all"` (options: "all", "young", "old", "train").
6. Implement the data loading logic inside `__init__`:
   - Wrap the loading in a `try/except Exception` block to gracefully handle missing files. If the file doesn't exist, log a warning and initialize an empty `self.samples = []` list so CI tests don't break.
   - Use `pd.read_hdf(h5_path, key=...)` to read three keys from the file: `'all features'`, `'trace metadata'`, and `'mouse metadata'`.
   - The data is stored with indices. Merge `'all features'` with `'trace metadata'` (joining on their common index/keys) to associate each run with a `mouse_id` and a chronological age (e.g., `age` or `age_months`).
   - Merge the result with `'mouse metadata'` (on `mouse_id`) to bring in the ultimate `lifespan` or time of death.
   - Filter the dataframe based on the `cohort` argument:
     - `young`: keep runs where the mouse age is <= 6 months (or equivalent in days).
     - `old`: keep runs where the mouse age is >= 24 months.
     - `train`: use a deterministic hash of `mouse_id` to select an 80% split of mice for training.
   - Extract the feature columns (excluding the metadata columns) and convert them to a NumPy array.
   - Apply z-score normalization to the features globally (handle division by zero and fill NaNs with 0.0).
   - Group the joined dataframe by `mouse_id` and sort each group chronologically.
   - Chunk each mouse's longitudinal trajectory into non-overlapping sequences of length `sequence_length`.
   - Store each chunk in `self.samples` as a dictionary: `{'trajectory_chunk': torch.tensor(chunk, dtype=torch.float32), 'chronological_age': ..., 'ultimate_lifespan': ...}`.
7. Implement `__len__(self)` returning `len(self.samples)`.
8. Implement `@jaxtyped(typechecker=beartype)` for `__getitem__(self, idx: int) -> Tensor:`. 
   - Ensure it returns ONLY `self.samples[idx]['trajectory_chunk']`. It must return just the raw sequence tensor so it plugs cleanly into our existing `JAXDictDataset` wrapper.
