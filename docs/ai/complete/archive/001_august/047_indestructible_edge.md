Context: We are continuing the consolidation of the Project MELD repository into
our "Director's Cut" master demonstrations. This task builds "Master Demo 2: The
Indestructible Edge", aimed directly at wet-lab researchers (e.g., UCSC
Braingeneers, Pașca Lab). It proves that our continuous-time State Space
architecture can survive the chaotic physical reality of laboratory
environments: it mathematically vetos massive environmental vibrations (fluidic
pumps) and dynamically routes around catastrophic hardware failures (dead
sensors) without dropping the sequence or reverting to discrete time.

Task: Create a new script `src/demo/02_indestructible_edge.py` by synthesizing
the orthogonal noise veto logic from `src/demo/1_hssm_veto_proof.py` and the
dynamic `NaN` sensor imputation logic from
`src/demo/9_dynamic_sensor_masking.py`.

Requirements:

1. **Imports & Setup:**

   - Import `torch`, `torch.nn`, `torch.optim`, `torch.nn.functional as F`,
     `numpy`, `os`, and `matplotlib.pyplot` (using `matplotlib.use("Agg")`).
   - Import `get_optimal_device` from `src.utils.device`.
   - Setup basic console logging (INFO level).

2. **The Architecture (`DynamicMaskingEngine`):**

   - Implement the `DynamicMaskingEngine` PyTorch class.
   - It must double the input dimension (`input_dim * 2`) via an initial
     `mask_encoder` (Linear -> LayerNorm -> GELU -> Linear) to encode the
     explicit `isnan` boolean mask.
   - Include a fallback `DemoSSM` class (using Causal Conv1d + Linear) exactly
     as implemented in `1_hssm_veto_proof.py` so the demo runs flawlessly on
     Mac/MPS without relying on `mamba_ssm` CUDA compilation. If
     `mamba_ssm.Mamba2` is available, use it; otherwise use `DemoSSM`.
   - The `forward` pass must: a) Detect NaNs: `mask = torch.isnan(x).float()` b)
     Sanitize to zero: `x_safe = torch.nan_to_num(x, nan=0.0)` c) Concatenate:
     `torch.cat([x_safe, mask], dim=-1)` d) Pass through `mask_encoder`, the SSM
     backbone, and a `predictor` linear head predicting `T+1`. e) Return
     `preds, mask`.

3. **The Data Generator (`WetLabDisasterSimulator`):**

   - Create a lightweight data generation class that yields a simulated 114-D
     sequence tensor `[Batch, Time, 114]`.
   - **Base Biology:** Generate a clean biological signal combining 5 distinct
     sine waves (frequencies between 0.5Hz and 3.0Hz) projected through a fixed
     random `[5, 114]` mixing matrix. Add standard normal noise (std=0.1).
   - **The Pump Artifact:** Inject a massive, continuous 2Hz sine wave
     (amplitude=5.0) across all features to simulate a mechanical fluidic pump
     vibration (the "Drowning Signal").
   - **Scenario "training":** Randomly drop 15% of the sensors to `float('nan')`
     for the entire sequence to teach the network spatial covariance. Return
     `corrupt_tensor` (with pump + NaNs) and `clean_tensor` (pure biology).
   - **Scenario "disaster":** At `DROP_FRAME = 100`, permanently kill 15% of the
     sensors (e.g., indices 90 to 110, representing a severed voltage array) by
     setting them to `float('nan')` for the remainder of the sequence. Return
     `corrupt_tensor`, `clean_tensor`.

4. **The Execution Flow (`main` function):**

   - Initialize device using `get_optimal_device(allow_mps=False, verbose=True)`
     (MPS may have issues with backward pass on some ops, so CPU/CUDA is safer).
   - Print a stark console header:
     `[*] BOOTING MASTER DEMO 2: THE INDESTRUCTIBLE EDGE`.
   - Instantiate the simulator and the `DynamicMaskingEngine`.
   - **The Burn-In (Training):** Run a fast 30-iteration loop (`AdamW`,
     lr=1e-3). In each iteration, get a training batch. Predict `T+1` on the
     `corrupt_tensor` and calculate `F.mse_loss` against the _uncorrupted,
     artifact-free_ `clean_tensor[:, 1:, :]` ground truth.
   - _Physics Note:_ This mathematically forces the SSM to simultaneously
     orthogonalize (veto) the 2Hz pump noise AND learn deep spatial covariance
     to impute the dropped sensors.
   - **The Disaster (Inference):** Call `generate_batch(scenario="disaster")`.
     Pass the `corrupt_tensor` through the engine in `eval()` mode.

5. **The Publication-Ready Dashboard:**
   - Create
     `plot_indestructible_dashboard(corrupt_seq, true_seq, pred_seq, drop_frame)`.
     Save to `output/02_indestructible_edge.png` using the `dark_background`
     theme.
   - 3-Panel Vertical Layout (figsize=12, 12, sharex=True):
     - **Panel 1 (The Input):** 2D Heatmap of the corrupted `test_seq`
       (transposed). Show the massive vertical banding of the pump artifact and
       the hard red vertical dashed line where the sensors drop to `NaN`. Title:
       "Analog Biological Layer (Massive Pump Artifact + Catastrophic Sensor
       Dropout)".
     - **Panel 2 (The Imputation & Veto):** Line plot focusing on one of the
       dropped sensors (e.g., index 100). Plot the _Clean Ground Truth_ biology
       (Cyan) vs the _Model's Imputed Prediction_ (Orange Dashed). The
       prediction should tightly track the pure biology, mathematically ignoring
       the 5.0 amplitude pump noise, even _after_ the sensor is physically
       severed at `DROP_FRAME`. Title: "State-Space Recovery: Artifact Vetoed &
       Missing Sensor Imputed".
     - **Panel 3 (The Error):** Line plot of the Absolute Error between the
       prediction and the clean ground truth for that sensor. Add a horizontal
       line representing the theoretical biological noise floor (e.g., 0.1).
       Title: "Imputation Error (Absolute Difference)".

Constraints:

- Keep the script fully self-contained so a user can run
  `python src/demo/02_indestructible_edge.py` and immediately see the physics
  proof.
- Ensure the console logging narrates the story (e.g.,
  `[*] Injecting 2Hz microfluidic pump vibration...`,
  `[*] Simulating catastrophic sensor failure (NaN)...`).
