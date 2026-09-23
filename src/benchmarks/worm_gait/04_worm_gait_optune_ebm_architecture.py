import os
import argparse
import yaml
import tempfile
import logging
import jax
import torch
import json
from torch.utils.data import DataLoader
import importlib
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import copy
import wandb
from src.benchmarks.worm_gait.core import run_worm_gait_experiment, build_graph
from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.data.datasets import JAXDictDataset
from src.echo.primitives.ebm import PrecisionWeightedEBM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
# Suppress noisy logs from the echo runner
logging.getLogger("src.echo.harness.echo_runner").setLevel(logging.WARNING)


def objective(trial, train_young_loader, eval_young_loader, eval_old_loader, base_config):
    # Suggest hyperparameters
    macro_hidden_size = trial.suggest_categorical(
        "observer.macro.ebm_hidden_size", [16, 32, 64, 128]
    )
    micro_hidden_size = trial.suggest_categorical("observer.micro.ebm_hidden_size", [128, 256])
    macro_depth = trial.suggest_categorical("observer.macro.ebm_depth", [2, 3])
    micro_depth = trial.suggest_categorical("observer.micro.ebm_depth", [2, 3])
    max_epochs = trial.suggest_categorical("optimization.max_epochs", [50, 100])
    learning_rate = trial.suggest_categorical("optimization.learning_rate", [1e-5, 5e-5, 1e-4])

    # Deep copy base config
    config = copy.deepcopy(base_config)

    # Update config
    config["observer"]["macro"]["ebm_hidden_size"] = macro_hidden_size
    config["observer"]["micro"]["ebm_hidden_size"] = micro_hidden_size
    config["observer"]["macro"]["ebm_depth"] = macro_depth
    config["observer"]["micro"]["ebm_depth"] = micro_depth
    config["optimization"]["max_epochs"] = max_epochs
    config["optimization"]["learning_rate"] = learning_rate

    # Create temporary config file for EchoRunner to read
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        tmp_path = tmp.name

    try:
        seed = config.get("experiment", {}).get("seed", 42)
        key = jax.random.PRNGKey(seed)
        key, kB = jax.random.split(key)

        # We run the experiment for PrecisionWeightedEBM only
        metrics, trace_young, trace_old, graph = run_worm_gait_experiment(
            config=config,
            ebm_class=PrecisionWeightedEBM,
            key=kB,
            train_young_loader=train_young_loader,
            eval_young_loader=eval_young_loader,
            eval_old_loader=eval_old_loader,
            config_path=tmp_path,
        )

        cohens_d = metrics["cohens_d"]
        wasserstein = metrics["wasserstein_distance"]

        # Maximize the absolute value of cohens_d + wasserstein_distance
        score = abs(cohens_d) + wasserstein

        return score
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    config_path = "configs/worm_gait_experiments.yaml"
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    benchmark_module = importlib.import_module("src.benchmarks.worm_gait.05_worm_gait_aging_ebm")

    seed = base_config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    # Load dataset once
    seq_len = base_config["dataset"]["ebm_seq_len"]
    try:
        train_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TRAIN.ts", seq_len=seq_len, is_aged=False
        )
        eval_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=False
        )
        eval_old_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=True
        )
        logger.info("Loaded RealEigenwormDataset")
    except FileNotFoundError:
        logger.warning("Local biological data not found. Falling back to SyntheticWormMockDataset.")
        train_young_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
        eval_young_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
        eval_old_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)

    # We need d_state which can be calculated using GaussianEBM or PrecisionWeightedEBM
    GaussianEBM = benchmark_module.GaussianEBM
    _, d_state = build_graph(GaussianEBM, key, base_config)

    train_young_dataset = JAXDictDataset(train_young_dataset_raw, d_state)
    eval_young_dataset = JAXDictDataset(eval_young_dataset_raw, d_state)
    eval_old_dataset = JAXDictDataset(eval_old_dataset_raw, d_state)

    batch_size = base_config.get("dataset", {}).get("batch_size", 2)
    train_young_loader = DataLoader(train_young_dataset, batch_size=batch_size, shuffle=True)
    eval_young_loader = DataLoader(eval_young_dataset, batch_size=batch_size, shuffle=False)
    eval_old_loader = DataLoader(eval_old_dataset, batch_size=batch_size, shuffle=False)

    study = optuna.create_study(direction="maximize", study_name="worm_gait_aging_ebm")

    # 1 hour timeout limit
    timeout_seconds = 3600

    wandb_kwargs = {"project": "worm_gait"}
    wandbc = WeightsAndBiasesCallback(metric_name="score", wandb_kwargs=wandb_kwargs)

    n_trials = base_config.get("experiment", {}).get("optuna_n_trials", 2)
    logger.info(f"Starting Optuna search with timeout of {timeout_seconds} seconds and max {n_trials} trials")
    study.optimize(
        lambda trial: objective(
            trial, train_young_loader, eval_young_loader, eval_old_loader, base_config
        ),
        n_trials=n_trials,
        timeout=timeout_seconds,
        catch=(Exception,),
        callbacks=[wandbc]
    )

    logger.info(f"Best Trial: {study.best_trial.value}")
    logger.info(f"Best Params: {study.best_trial.params}")

    # Save best parameters to a JSON for easy extraction later
    os.makedirs("output/benchmarks/worm_gait", exist_ok=True)
    with open("output/benchmarks/worm_gait/04_worm_gait_ebm_best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
        
    wandb.summary["best_params"] = study.best_trial.params
    artifact = wandb.Artifact("04_optune_best_params", type="metrics")
    artifact.add_file("output/benchmarks/worm_gait/04_worm_gait_ebm_best_params.json")
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
