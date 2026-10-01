**Context Files to Load / Modify:**
* `src/echo/benchmarks/06_worm_gait_decline.py`
* `src/echo/benchmarks/07_insilico_reprogramming.py`
* `configs/echo_training.yaml`

**Task: Paper 1 Climax - Enable True Training and Connect the Pipeline**
We have successfully built the architecture, but the `06_worm_gait_decline.py` script is currently skipping the training phase. A randomly initialized model cannot carve a Waddington basin, which causes the ablation histograms to overlap and the in-silico rescue to look like Brownian motion. We must actually train the models, serialize the learned physics, and load them into script 07.

**Core Objectives (Script 06 - `06_worm_gait_decline.py`):**

**1. Enable Actual Training:**
* Remove the entire `tempfile.NamedTemporaryFile` block that generates the hardcoded YAML config with `max_epochs: 0`.
* Point `runner_A` and `runner_B` to the permanent configuration file: `EchoRunner("configs/echo_training.yaml")`. 
* Ensure `configs/echo_training.yaml` exists and has `max_epochs` set to a real training value (e.g., `100`). This will allow the `EchoTrainer` to physically carve the deep Waddington basin on the Young worms.

**2. Serialize the Trained Engine:**
* After `runner_B` finishes training the `PrecisionWeightedEBM` (Run B), we must save this specific model so the Rescue script can use it.
* Use Equinox serialization to save the trained weights:
  `eqx.tree_serialise_leaves("output/echo/trained_young_worm_engine.eqx", graph_B)`
* Ensure the `output/echo/` directory exists before saving.

**Core Objectives (Script 07 - `07_insilico_reprogramming.py`):**

**3. Load the Trained Engine:**
* Instantiate the skeleton `PredictiveCodingGraph` exactly as it was built in Run B of script 06 (utilizing the `PrecisionWeightedEBM`).
* Immediately after creating the skeleton `graph`, overwrite its weights by loading the trained artifact:
  `graph = eqx.tree_deserialise_leaves("output/echo/trained_young_worm_engine.eqx", graph)`
* *Fallback:* Wrap the deserialization in a `try/except Exception` block. If the file is missing or fails to load, log a warning (`logger.warning("Trained model not found! Proceeding with random initialization.")`) and continue with the random graph to prevent CI crashes.

**4. Execute the Rescue:**
* The rest of the SDE simulation logic in script 07 remains exactly the same. The only difference is that now, when you apply the `lambda_gain = 5.0` intervention, it is amplifying a *real* energy landscape, which should force the erratic trajectory to snap into the learned biological limit cycle.

**Constraints:**
* Strictly maintain Equinox functional purity. `eqx.tree_deserialise_leaves` requires a matching PyTree skeleton to populate.
* **Logging:** Keep logs peaceful and precise (e.g., `logger.info("Serializing trained Young Worm engine to disk.")`). No dramatic prints.
