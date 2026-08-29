# TASK 1: Build the Continuous Human Telemetry Dataloader

We need to transition the dataloaders from ingesting micro-scale cellular data
to ingesting macroscopic human wearable data.

Please create a new file: `src/pipeline/sim2real/human_telemetry_dataloader.py`

**Requirements:**

1. **Class Name:** `HumanTelemetryLoader` (following the pattern of
   `BioelectricLoader`).
2. **Inputs:** Mock the ingestion of continuous, multi-scale human wearable data
   simulating a middle-aged human.
   - **Continuous HRV:** 250Hz sampling (RR intervals / ECG proxy).
   - **Actigraphy (Movement):** 50Hz sampling (accelerometer data).
   - **Core Temperature:** 1Hz sampling.
   - **Continuous Glucose Monitor (CGM):** 1/60Hz sampling (1 sample every
     minute).
3. **Temporal Alignment:** Like the `phase_structure_dataloader.py`, use a
   `align_to_master_clock` method to align all variables to the 250Hz HRV master
   frequency. For slow variables like CGM, forward-fill the data but generate an
   accurate `mask` tensor (1.0 for present, 0.0 for missing) to feed into our
   `MaskAwareSSM`.
4. **Therapy Shock Method:** Add a method
   `apply_therapy_shock(pure_vitals, time_minutes)`. This should simulate the
   systemic stress ($D_0$) of ECM-breaking enzymes being administered via IV:
   HRV becomes highly erratic (variance spikes), Core Temp spikes (sterile
   inflammation), and movement drops to near zero.
5. **Output Shape:** The dataloader should yield a combined dictionary
   `{"x_raw": tensor, "mask": tensor}` of shape `[Time, Features]`.
