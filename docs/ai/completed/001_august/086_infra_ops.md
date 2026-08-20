We are executing Phase 3: Cloud MLOps & Distributed Scale (W&B + Ray).

Our goal is to wrap our unified runner in Ray Tune for parallel execution and deeply integrate Weights & Biases so all generated artifacts (PNG dashboards, JSON reports, CSVs) are automatically uploaded to the cloud.

Please execute the following steps carefully:

1. Install Dependencies
Add ray[tune] and wandb to requirements.txt.

2. Clean Up StableBasinTrainer (src/harness/trainer.py)
In train_epoch, ensure the W&B logging safely calls wandb.log() ONLY if a run is active. (Since wandb.init() will be handled by the Ray worker, the trainer just logs to the active run).

```python
if use_wandb:
    import wandb
    if wandb.run is not None:
        wandb.log({"train_loss": avg_loss, "epoch": epoch, "epoch_time": epoch_time})
```
3. Refactor src/harness/clinical_diagnostic_runner.py for Ray Tune
We need to convert our sequential loop into a Ray Tune trainable function and use tune.grid_search to distribute the models.

- Import Ray: import ray and from ray import tune, train.

- Import WandB: import wandb.

- Extract the inner execution logic (that runs a single model) into a new function: def evaluate_model(trial_config):.

    - Inside evaluate_model(trial_config):

    - Extract parameters: config = trial_config["base_config"] and model_type = trial_config["model_type"].

    - Get device: device = get_optimal_device(verbose=False).

    - Initialize W&B inside the worker so each model gets its own run in the dashboard:
    ```python
    wandb.init(
        project="stable-basin",
        name=f"diagnostic_{model_type}",
        config={"model_type": model_type, **config},
        reinit=True
    )
    ```
    - Run the dataset loading, model initialization, training (trainer.fit(dataloader, epochs, use_wandb=True)), and crash detection as normal.

    - Artifact Uploads: After generating and saving the PNG, JSON, and CSV locally to output/harness/, log them to W&B:
    ```python
    if wandb.run is not None:
    wandb.log({
        "inference_latency_ms": latency_ms,
        "baseline_ksm_variance": base_ksm_variance,
        "crash_frame": crash_frame,
        "dashboard": wandb.Image(png_path) # Uploads the PNG for UI viewing
    })

    artifact = wandb.Artifact(f"diagnostic_{model_type}", type="report")
    artifact.add_file(report_path)
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)
    ```
    - Report back to Ray Tune: train.report({"latency_ms": latency_ms, "crash_frame": crash_frame, "baseline_ksm_variance": base_ksm_variance}).

    - Close W&B: wandb.finish().

- In main():

    - Load the YAML config using yaml.safe_load.

    - Define the Ray Tune search space (a dictionary mapping model names to their configs).
    ```python
    search_space = {
    "base_config": config,
    "model_type": tune.grid_search(config["models"])
}
    ````
    
    - Initialize Ray and launch the Tuner:
    ```python
    ray.init(ignore_reinit_error=True)

    tuner = tune.Tuner(
        # Wrap the function to specify resources (e.g., 1 CPU or 1 GPU per model)
        tune.with_resources(
            evaluate_model, 
            resources={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
        param_space=search_space,
        run_config=train.RunConfig(name="clinical_diagnostic_sweep")
    )

    results = tuner.fit()
    logger.info("Ray Tune execution complete.")
    ```

4. Verification
Ask the user to run wandb login in their terminal (if they haven't already), and then execute make clinical-diagnostic. Verify that Ray spins up multiple workers, executes the models in parallel, and uploads the dashboards to the Weights & Biases UI!

What to expect when this finishes:
Once implemented, we will have true "push-button ML".
When you run make clinical-diagnostic, Ray will instantly spawn 4 separate worker processes. If your machine/cloud instance has multiple GPUs or sufficient CPU cores, they will train simultaneously.
