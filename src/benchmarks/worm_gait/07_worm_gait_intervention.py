from jaxtyping import PRNGKeyArray
import os
import logging
import argparse
import yaml
import json
import wandb
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import energy_distance

from src.benchmarks.aging_resilience.task_registry import get_benchmark_task
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph
from src.echo.primitives.ebm import PrecisionWeightedEBM

from src.benchmarks.worm_gait.core import setup_experiment, simulate_sde
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(graph, x0, key, config):
    N_steps = config["experiment"]["N_steps"]
    dt = config["experiment"]["dt"]
    num_runs = config["experiment"]["num_runs"]
    lambda_A = config["intervention"]["lambda_A"]
    lambda_B = config["intervention"]["lambda_B"]

    logger.info(f"Simulating Run A: Degraded baseline with precision_injection_gain={lambda_A}")
    keys_A = jax.random.split(key, num_runs)
    vmap_simulate = eqx.filter_jit(
        jax.vmap(simulate_sde, in_axes=(None, None, None, None, None, 0))
    )
    traj_A_batch = vmap_simulate(graph, x0, lambda_A, N_steps, dt, keys_A)

    logger.info(f"Simulating Run B: Therapeutic rescue with precision_injection_gain={lambda_B}")
    keys_B = jax.random.split(key, num_runs)  # Same seeds for fair comparison
    traj_B_batch = vmap_simulate(graph, x0, lambda_B, N_steps, dt, keys_B)

    return traj_A_batch, traj_B_batch


def calculate_metrics(graph, traj_A_batch, traj_B_batch, config):
    logger.info("Computing energy distance rescue metric R(lambda).")
    
    task = get_benchmark_task(config)
    _, young_eval_loader, _ = task.get_dataloaders(config, None, batch_size=8)
    
    Y_list = []
    for batch in young_eval_loader:
        if isinstance(batch, dict) and "s_true" in batch:
            Y_list.append(batch["s_true"].numpy().flatten())
        else:
            Y_list.append(batch.numpy().flatten())
    Y = np.concatenate(Y_list)

    d_internal = graph.hull.d_internal
    d_sensory = graph.hull.d_sensory

    def get_sensory_flat(traj_batch):
        # traj_batch shape: [num_runs, N_steps + 1, d_full]
        # Extract the sensory portion of the state
        sensory = traj_batch[:, :, d_internal : d_internal + d_sensory]
        return np.array(sensory).flatten()
    
    M_lambda_A = get_sensory_flat(traj_A_batch)
    M_lambda_B = get_sensory_flat(traj_B_batch)
    
    dist_A = float(energy_distance(Y, M_lambda_A))
    dist_B = float(energy_distance(Y, M_lambda_B))
    
    # Calculate R(lambda_B)
    R_lambda_B = 1.0 - (dist_B / dist_A) if dist_A != 0 else 0.0

    return dist_A, dist_B, R_lambda_B


def plot_results(traj_A, traj_B, dist_A, dist_B, R_lambda_B, config):
    logger.info("Generating Figure 4 Beacon Plot.")
    output_plot = config["paths"]["output_plot"]
    lambda_A = config["intervention"]["lambda_A"]
    lambda_B = config["intervention"]["lambda_B"]

    lambda_A_str = f"{lambda_A:.2f}"
    lambda_B_str = f"{lambda_B:.2f}"

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig = plt.figure(figsize=(18, 6))

    # Panel A: The Pathology
    ax1 = fig.add_subplot(131, projection="3d")
    tA_np = np.array(traj_A)
    ax1.plot(tA_np[:, 0], tA_np[:, 1], tA_np[:, 2], color="red", alpha=0.7, linewidth=1)
    ax1.scatter(tA_np[0, 0], tA_np[0, 1], tA_np[0, 2], color="black", s=50, label="x0 (Synthetically Degraded)")
    ax1.set_title(f"Panel A: Degraded Pathology (λ={lambda_A_str})")
    ax1.legend()

    # Panel B: The Phase Space Rescue
    ax2 = fig.add_subplot(132, projection="3d")
    tB_np = np.array(traj_B)
    ax2.plot(tB_np[:, 0], tB_np[:, 1], tB_np[:, 2], color="green", alpha=0.7, linewidth=1)
    ax2.scatter(tB_np[0, 0], tB_np[0, 1], tB_np[0, 2], color="black", s=50, label="x0 (Synthetically Degraded)")
    ax2.set_title(f"Panel B: Therapeutic Rescue (λ={lambda_B_str})")
    ax2.legend()

    # Panel C: Energy Distance Metric
    ax3 = fig.add_subplot(133)
    
    metrics_labels = [f"Baseline (λ={lambda_A_str})", f"Rescue (λ={lambda_B_str})"]
    R_values = [0.0, R_lambda_B]
    
    ax3.bar(metrics_labels, R_values, color=['gray', 'green'], alpha=0.7)
    
    ax3.set_ylabel(r"Therapeutic Rescue $R(\lambda)$")
    ax3.set_title("Panel C: Energy Distance Restoration")
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(w_pad=4.0)
    plt.savefig(output_plot, dpi=300)
    plt.close()

    logger.info(f"Figure 4 successfully generated and saved to: {output_plot}")


def save_results(
    dist_A, dist_B, R_lambda_B, config
):
    output_metrics = config["paths"]["output_metrics"]
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)

    data = {
        "config": config,
        "RunA": {
            "lambda": config["intervention"]["lambda_A"],
            "energy_distance": dist_A,
        },
        "RunB": {
            "lambda": config["intervention"]["lambda_B"],
            "energy_distance": dist_B,
            "R_lambda": R_lambda_B
        }
    }

    with open(output_metrics, "w") as f:
        json.dump(data, f, indent=2)

    wandb.log({
        "energy_distance_baseline": dist_A,
        "energy_distance_rescue": dist_B,
        "R_lambda_rescue": R_lambda_B
    })
    wandb.log({"07_worm_gait_intervention_rescue": wandb.Image(config["paths"]["output_plot"])})
    
    artifact = wandb.Artifact("07_worm_gait_intervention_metrics", type="metrics")
    artifact.add_file(output_metrics)
    wandb.log_artifact(artifact)

    logger.info(f"Summary metrics successfully saved to: {output_metrics}")


def main():
    parser = argparse.ArgumentParser(description="Worm Gait Intervention Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/worm_gait_experiments.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Dynamically inject lambda_A from inferred baseline
    inferred_lambda_path = "output/benchmarks/worm_gait/06_inferred_biological_lambda.json"
    if os.path.exists(inferred_lambda_path):
        with open(inferred_lambda_path, "r") as f:
            lambda_data = json.load(f)
            inferred_lambda = lambda_data.get("biological_lambda")
            if inferred_lambda is not None:
                config["intervention"]["lambda_A"] = inferred_lambda
    else:
        logger.warning(f"Inferred lambda not found at {inferred_lambda_path}, falling back to config.")

    wandb.init(project="stable_basin_aging", group=config.get("dataset", {}).get("name", "worm_gait"), name="07_worm_gait_intervention", config=config)
    
    # Establish lineage
    weights_path = config["paths"]["model_weights"]
    wandb.run.use_artifact("05_worm_gait_decline_trained_engine:latest", type="model")

    graph, x0, key = setup_experiment(config)

    traj_A_batch, traj_B_batch = run_experiment(graph, x0, key, config)

    dist_A, dist_B, R_lambda_B = calculate_metrics(graph, traj_A_batch, traj_B_batch, config)

    plot_results(
        traj_A_batch[0],
        traj_B_batch[0],
        dist_A,
        dist_B,
        R_lambda_B,
        config,
    )

    save_results(
        dist_A,
        dist_B,
        R_lambda_B,
        config,
    )
    wandb.finish()

if __name__ == "__main__":
    main()
