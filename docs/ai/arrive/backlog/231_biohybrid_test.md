Unit Test Suite for Biohybrid Dataset
Context Files to Load:

src/data/ephys/synthetic_biohybrid.py

tests/conftest.py (if present)

Prompt Text:

Plaintext
Create a unit test suite at tests/data/test_synthetic_biohybrid.py to validate the SyntheticBiohybridDataset.

Requirements:
1. Implement test cases using pytest:
   - `test_dataset_shapes_and_types`: Verify that output tensors (`wideband`, `lfp`, `spikes`, `optogenetic_actuation`, `mask`) match expected shapes [seq_len, num_channels] and dtype torch.float32.
   - `test_spike_field_coherence`: Verify that spike occurrences correlate significantly with preferred LFP phases rather than being purely uniform noise.
   - `test_optogenetic_evoked_response`: Verify that when an optical write pulse is injected, the spike count in the subsequent 2 ms window increases significantly compared to baseline.
   - `test_reproducibility`: Verify that passing a fixed seed produces deterministic tensors.
   - `test_multichannel_scaling`: Test initialization with 64, 128, and 512 channels without memory leaks or shape mismatches.

2. Coding Standards:
   - Keep test assertions clean and standard.
   - Ensure test suite passes natively via `pytest tests/data/test_synthetic_biohybrid.py` without requiring special environment flags.
