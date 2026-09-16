# Weights & Biases (WandB) Integration Plan

This document outlines the architecture for migrating the Stable Basin local
training harnesses (`sensor_fusion_runner` and `density_sweep_runner`) to a
production-grade ML tracking and orchestration setup using **Weights & Biases
(WandB)**.

## Goal

To replace the localized file-based logging and sequential `Makefile` execution
with a cloud-native, distributed pipeline that provides real-time dashboarding,
hyperparameter tracking, and artifact management.

## User Review Required

> [!IMPORTANT] **API Key Setup:** You will need to create a free account at
> [wandb.ai](https://wandb.ai) and run `wandb login` on your machine to
> authenticate the runners before executing this plan.
>
> **Do you want to proceed with just the WandB experiment tracking for now, or
> should we also implement a full `Ray Tune` / `WandB Sweeps` distributed
> orchestration layer?** (For now, the plan focuses on tracking).

## Open Questions

1. **Project Naming:** Do you want all runs logged under a single WandB project
   (e.g., `stable-basin`), or separated by task (e.g.,
   `stable-basin-sensor-fusion` and `stable-basin-density-sweep`)?
2. **Offline Mode:** Do you want the ability to run these offline and sync the
   logs to WandB later, or require an active internet connection during
   training?

## Proposed Changes

---

### Harness Runners (Experiment Tracking)

The core logic of the training loops will remain identical, but we will augment
the existing `logger.info` and `timing_log.write` calls with `wandb.log()`.

#### [MODIFY] [sensor_fusion_runner.py](file:///Users/ry/gh/stable-basin/src/harness/sensor_fusion_runner.py)

- Add a `--wandb` toggle to the `argparse` configuration to enable/disable
  syncing.
- Initialize the run via `wandb.init(project="stable-basin", config=args)`.
- In the `for batch in dataloader:` loop, log per-batch and per-epoch metrics:
  `wandb.log({"epoch": epoch, f"{name}/loss": loss, f"{name}/time_per_batch": batch_time})`.
- At the end of the evaluation phase, log the final OOD-MSE scores.
- Upload the generated `01_baseline_interpolation.csv` and `.png` plots directly
  to the WandB dashboard using
  `wandb.log({"dashboard": wandb.Image(png_path)})`.

#### [MODIFY] [density_sweep_runner.py](file:///Users/ry/gh/stable-basin/src/harness/density_sweep_runner.py)

- Initialize `wandb.init()` with sweep configurations (density, seed, model).
- Log the OOD-MSE score for each sweep configuration.
- Log the generated `03_sensor_density_sweep.png` statistical rigor graph to the
  cloud dashboard.

---

### Configuration & Orchestration (WandB Sweeps)

To stop running models sequentially, we will introduce a WandB Sweep
configuration that allows multiple workers (GPUs/Machines) to pull jobs from a
central cloud queue.

#### [NEW] [sweep.yaml](file:///Users/ry/gh/stable-basin/sweep.yaml)

- Create a YAML configuration defining a grid search across:
  - `models`: [baseline, gru_d, ode_rnn, mask_aware]
  - `seeds`: [42, 100, 256, 512]
  - `densities`: [0.1, 0.05, 0.01, 0.001]
- **How it works:** Instead of a `Makefile`, you launch `wandb agent sweep_id`
  on as many machines as you want, and they will chew through the permutations
  in parallel.

---

## Verification Plan

### Automated Tests

- None explicitly needed for the tracking layer, but existing Pytest suites will
  be run to ensure we didn't break PyTorch imports.

### Manual Verification

1. Run
   `python -m src.harness.sensor_fusion_runner --epochs 1 --train-seq-len 10 --test-seq-len 10 --wandb`.
2. Provide the generated WandB URL to the user to visually inspect the live loss
   curves and artifacts on their cloud dashboard.
