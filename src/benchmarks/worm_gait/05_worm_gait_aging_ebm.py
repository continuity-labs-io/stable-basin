import os
import argparse
import yaml
import tempfile
import logging
import jax
import jax.numpy as jnp
import torch
import json
import wandb
import pingouin as pg
from scipy.stats import ks_2samp, wasserstein_distance
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx

from src.benchmarks.worm_gait.core import build_graph, get_full_states, compute_full_trace, compute_metrics, run_worm_gait_experiment
from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph
from src.data.datasets import JAXDictDataset
from src.echo.primitives.ebm import GaussianEBM, PrecisionWeightedEBM
from src.echo.harness.echo_runner import EchoRunner
from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.metrics.energy_landscape import batch_calculate_curvature

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)



def plot_ablation_results(
    eval_young_dataset_raw,
    eval_old_dataset_raw,
    trace_young_A,
    trace_old_A,
    trace_young_B,
    trace_old_B,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    traj_young = eval_young_dataset_raw.data[0].numpy()
    traj_old = eval_old_dataset_raw.data[0].numpy()
    axes[0].plot(traj_young[:500, 0], traj_young[:500, 1], label="Young (Day 1-3)")
    axes[0].plot(traj_old[:500, 0], traj_old[:500, 1], label="Old (Day 9+)", alpha=0.7)
    axes[0].set_title("Panel A: The Limit Cycle")
    axes[0].set_xlabel("Sensor Dimension 0")
    axes[0].set_ylabel("Sensor Dimension 1")
    axes[0].legend()
    trace_young_A_np = np.nan_to_num(np.array(trace_young_A), nan=1.0)
    trace_old_A_np = np.nan_to_num(np.array(trace_old_A), nan=1.0)
    trace_young_B_np = np.nan_to_num(np.array(trace_young_B), nan=1.0)
    trace_old_B_np = np.nan_to_num(np.array(trace_old_B), nan=1.0)

    # Panel B: Laplace Flatline We use a thick line for Young and a dashed line
    # for Old because the GaussianEBM's Hessian is mathematically constant
    # across the state space, causing perfect overlap.
    axes[1].plot(trace_young_A_np, label="Young", color="blue", linewidth=4)
    axes[1].plot(trace_old_A_np, label="Old", color="orange", linestyle="--", linewidth=2)
    axes[1].set_title("Panel B: Laplace Flatline")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Hessian Trace (Curvature)")
    axes[1].legend()

    # Panel C: Waddington Basin Flattening
    axes[2].hist(trace_young_B_np, bins=20, alpha=0.5, label="Young", color="blue", density=True)
    axes[2].hist(
        trace_old_B_np,
        bins=20,
        alpha=0.7,
        label="Old",
        color="orange",
        density=True,
        histtype="step",
        linewidth=2,
    )
    axes[2].set_title("Panel C: Waddington Basin Flattening")
    axes[2].set_xlabel("Hessian Trace (Curvature)")
    axes[2].set_ylabel("Density")
    axes[2].legend()

    plt.tight_layout()
    os.makedirs("output/benchmarks/worm_gait", exist_ok=True)
    plt.savefig("output/benchmarks/worm_gait/05_worm_gait_decline_ablation.png")
    
    if wandb.run is not None:
        wandb.log({"05_worm_gait_decline_ablation": wandb.Image("output/benchmarks/worm_gait/05_worm_gait_decline_ablation.png")})
    plt.close()

    logger.info(
        "Benchmark complete. Plot saved to output/benchmarks/worm_gait/05_worm_gait_decline_ablation.png"
    )



def main():
    parser = argparse.ArgumentParser(description="Worm Gait Aging EBM Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/worm_gait_experiments.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Young (Train) and Old (Eval) datasets.")
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    try:
        seq_len = config["dataset"]["ebm_seq_len"]
        train_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TRAIN.ts", seq_len=seq_len, is_aged=False
        )
        eval_young_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=False
        )
        eval_old_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=True
        )
    except FileNotFoundError:
        logger.warning("Local biological data not found. Falling back to SyntheticWormMockDataset.")
        train_young_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
        eval_young_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)
        eval_old_dataset_raw = SyntheticWormMockDataset(seq_len=seq_len, num_samples=50)

    # Determine d_state
    _, d_state = build_graph(GaussianEBM, key, config)

    train_young_dataset = JAXDictDataset(train_young_dataset_raw, d_state)
    eval_young_dataset = JAXDictDataset(eval_young_dataset_raw, d_state)
    eval_old_dataset = JAXDictDataset(eval_old_dataset_raw, d_state)

    batch_size = config.get("dataset", {}).get("batch_size", 2)
    train_young_loader = DataLoader(train_young_dataset, batch_size=batch_size, shuffle=True)
    eval_young_loader = DataLoader(eval_young_dataset, batch_size=batch_size, shuffle=False)
    eval_old_loader = DataLoader(eval_old_dataset, batch_size=batch_size, shuffle=False)

    key, kA = jax.random.split(key)
    metrics_A, trace_young_A, trace_old_A, graph_A = run_worm_gait_experiment(
        config, GaussianEBM, kA, train_young_loader, eval_young_loader, eval_old_loader, args.config
    )

    key, kB = jax.random.split(key)
    metrics_B, trace_young_B, trace_old_B, graph_B = run_worm_gait_experiment(
        config,
        PrecisionWeightedEBM,
        kB,
        train_young_loader,
        eval_young_loader,
        eval_old_loader,
        args.config,
    )

    logger.info("Serializing trained Young Worm engine to disk.")
    os.makedirs("output/benchmarks/worm_gait", exist_ok=True)
    eqx.tree_serialise_leaves(
        "output/benchmarks/worm_gait/05_worm_gait_decline_trained_engine.eqx", graph_B
    )

    all_metrics = {"GaussianEBM": metrics_A, "PrecisionWeightedEBM": metrics_B}

    metrics_path = "output/benchmarks/worm_gait/05_worm_gait_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Serialized full statistical metrics to {metrics_path}")

    plot_ablation_results(
        eval_young_dataset_raw,
        eval_old_dataset_raw,
        trace_young_A,
        trace_old_A,
        trace_young_B,
        trace_old_B,
    )
    
    artifact = wandb.Artifact("05_worm_gait_decline_trained_engine", type="model")
    artifact.add_file("output/benchmarks/worm_gait/05_worm_gait_decline_trained_engine.eqx")
    wandb.log_artifact(artifact)
    
    metrics_artifact = wandb.Artifact("05_worm_gait_metrics", type="metrics")
    metrics_artifact.add_file(metrics_path)
    wandb.log_artifact(metrics_artifact)
    
    wandb.finish()


if __name__ == "__main__":
    main()
