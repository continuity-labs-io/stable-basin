# Ephys Data Ingestion Plan

In the `data/ephys` directory we have 3 types of raw, unstructured
electrophysiology data (detailed in `/src/data/ephys/data_options.md`):

1. **FinalSpark Whole-Life Telemetry**
   (`data/ephys/finalspark/fs437_export/fs437_raw.hdf5`)
2. **Pharmacological Shock** (`data/ephys/pharmacological_shock/*.raw.h5`)
3. **3Brain HD-MEA** (`data/ephys/hdmea_neuropulse.brw`)

We must create the data ingestion for these raw messy data sets to try out each
data type.

## Architectural Decision: 3 Distinct PyTorch Datasets

Yes, we absolutely need to create **3 distinct PyTorch Dataset classes**.
Because these files are massive (ranging from 1.7GB to over 60GB) and originate
from completely different hardware arrays (which means different internal
HDF5/BRW tree structures), they require tailored ingestion logic.

The most critical requirement is that these classes must **lazy-load** the data.
They cannot load the full arrays into memory (which would cause an instant OOM
crash). Instead, they must map the length of the dataset in `__init__` and slice
chunks directly from the on-disk HDF5 file inside `__getitem__`.

### 1. FinalSpark Dataset

- **File**: `src/data/ephys/finalspark_dataset.py`
- **Class**: `FinalSparkDataset`
- **Mechanism**: Opens `fs437_raw.hdf5` to stream high-frequency continuous
  electrical telemetry.
- **Documentation**: Will migrate the "FinalSpark Whole-Life Telemetry" origin,
  sampling, and MVM Proof justification from `data_options.md` directly into the
  class docstring.

### 2. Pharmacological Shock Dataset

- **File**: `src/data/ephys/pharma_shock_dataset.py`
- **Class**: `PharmacologicalShockDataset`
- **Mechanism**: The `__init__` will accept a drug condition (e.g., `"control"`,
  `"10uM"`) to dynamically target the correct `.raw.h5` file, allowing us to
  test phase transitions.
- **Documentation**: Will migrate the "Pharmacological Shock" origin, sampling,
  and phase transition MVM Proof justification from `data_options.md` directly
  into the class docstring.

### 3. HD-MEA Dataset

- **File**: `src/data/ephys/hdmea_dataset.py`
- **Class**: `HDMEADataset`
- **Mechanism**: Parses the specialized BrainWave (`.brw`) HDF5 container to
  stream the massive 4,096-channel spatial grid.
- **Documentation**: Will migrate the "3Brain HD-MEA 4,096-Channel Stress Test"
  origin, sampling, and hardware scale MVM Proof justification from
  `data_options.md` directly into the class docstring.

## Multiprocessing Safety

PyTorch's `DataLoader` using multiple workers (`num_workers > 0`) can cause
segmentation faults when passing a single open `h5py.File` handle across
processes.

To solve this, we will instantiate the HDF5 file handle _inside_ `__getitem__`
(opening and closing it per fetch) or use a robust `worker_init_fn`. Opening it
inside `__getitem__` is the safest, most robust method for streaming from disk
in PyTorch, despite a marginal overhead.
