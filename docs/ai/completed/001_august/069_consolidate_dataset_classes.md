# Dataloader Unification Plan

Currently, we have PyTorch Dataset classes scattered across `src/pipeline/` and `src/data/`. As the project scales, having datasets in a `pipeline` module creates architectural confusion. The standard convention is to consolidate all `torch.utils.data.Dataset` implementations strictly under `src/data/`. 

## 1. Directory Migration
We will move all dataset loaders from `src/pipeline/` into `src/data/`:

- **Ephys Datasets**:
  - `src/pipeline/ephys/brw_dataloader.py` $\to$ `src/data/ephys/`
  - `src/pipeline/ephys/maxwell_dataloader.py` $\to$ `src/data/ephys/`
  - `src/pipeline/ephys/spike_ingestion.py` $\to$ `src/data/ephys/`
  - `src/pipeline/ephys/uhd_lfp_dataloader.py` $\to$ `src/data/ephys/`

- **Optical Datasets**:
  - `src/pipeline/optical/aollsm_dataloader.py` $\to$ `src/data/optical/`

- **Simulated Datasets**:
  - `src/pipeline/neocortical_assembloid_dataloader.py` $\to$ `src/data/simulators/`

## 2. Duplicate Resolution
We recently created `src/data/ephys/hdmea_dataset.py` which loads `.brw` files. However, `src/pipeline/ephys/brw_dataloader.py` also exists and loads `.brw` files (via `ContinuousHDMEADataset`). 
- **Action**: We will permanently delete `ContinuousHDMEADataset` (`src/pipeline/ephys/brw_dataloader.py`) as it is now obsolete. We will update `src/demo/raw/8_ephys_demo.py` to import and utilize the newly created `HDMEADataset` (`src/data/ephys/hdmea_dataset.py`) instead.

## 3. Global Import Refactoring
After migrating the files, we must run a global search-and-replace across the entire codebase to update the import paths.
For example, updating:
`from src.pipeline.ephys.maxwell_dataloader import MaxWellHDMEADataset`
to:
`from src.data.ephys.maxwell_dataloader import MaxWellHDMEADataset`

This will affect multiple files in `src/demo/raw/`, `src/demo/`, and potentially test scripts.

## 4. Pipeline Cleanup
Once the datasets are safely inside `src/data/`, we will evaluate what remains in `src/pipeline/` (e.g., `credentials.json`, `sim2real/`). If `src/pipeline` is only holding datasets, we can delete the directory entirely to keep the root tree clean.
