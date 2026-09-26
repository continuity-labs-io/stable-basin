Create a new data engineering script at `tools/build_killifish_dataset.py`. This script will parse the raw African turquoise killifish dataset extracted from Zenodo and compile it into a PyTorch-ready Dataset for continuous-time physics modeling.

### Context & Directory Structure
Assume the Zenodo archive `data.zip` has been extracted to `data/killifish/`. 
Based on the repository documentation, the files we need are located here:
1. Ground Truth Metadata: `data/killifish/data/a1_20241119/26441580/df_reformat_10_20241119.csv`. This contains the `lifespan`, `age_days`, `status`, and `fish_number`.
2. Continuous Kinematics: `data/killifish/data/p3_20230526/test/standardization/`. This directory contains subfolders for individual fish, filled with `.h5` files containing 20Hz pose features.

### Requirements for `tools/build_killifish_dataset.py`:
1. **Metadata Parsing:** 
   - Load the metadata CSV using pandas.
   - Filter the metadata to only include fish that died of natural causes (`status == 'd'`).
   - Create a mapping dictionary linking each `fish_number` to its absolute `lifespan`.

2. **Kinematic Data Extraction:**
   - Use `pathlib` to iterate through the `standardization` directory.
   - For each valid `.h5` file, load the 57 continuous pose features (stored as columns ending in `_m` or `_s`). 
   - Ensure you skip any columns containing metadata (like `frame_count` or `frame_timestamp`).
   - Normalize the continuous features (mean zero, unit variance) to ensure stable gradients in the physics engine downstream.

3. **PyTorch Dataset Abstraction:**
   - Create a PyTorch `Dataset` class named `KillifishContinuousDataset`.
   - The `__getitem__` method should return a tuple: `(trajectory_chunk, chronological_age, ultimate_lifespan)`.
   - Implement a chunking mechanism (e.g., `sequence_length=100`) so the 20Hz data is sliced into tractable continuous-time windows for the Energy-Based Model.

4. **Logging & Professional Standards:**
   - Dial down the intensity of all logging. Use standard `logging` with subdued, precise language (e.g., `logger.info("Parsed continuous features for fish ID %s.", fish_id)`). Do not use emojis, exclamation marks, or dramatic formatting.
   - If novel software library dependencies are required (e.g., `h5py`, `pandas`), output a dedicated 'Dependencies' section at the end of your response.

Please output the complete script.

