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
import pingouin as pg
from scipy.stats import ks_2samp, wasserstein_distance

from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph
from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.metrics.energy_landscape import batch_calculate_curvature

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
    logger.info("Computing thermodynamic restoration metrics (Hessian trace).")
    energy_fn = lambda x: graph.ebm(x)[0]
    num_runs = config["experiment"]["num_runs"]
    lambda_A = config["intervention"]["lambda_A"]
    lambda_B = config["intervention"]["lambda_B"]

    def get_traces(traj_batch):
        traces = []
        for i in range(num_runs):
            metrics = batch_calculate_curvature(energy_fn, traj_batch[i])
            traces.append(np.array(metrics["hessian_trace"]))
        return np.vstack(traces)

    trace_A_batch = get_traces(traj_A_batch)
    trace_B_batch = get_traces(traj_B_batch)

    if np.isnan(trace_A_batch).any():
        logger.warning(f"NaNs detected in Run A (lambda={lambda_A}) trace. Imputing with 0.0.")
        trace_A_batch = np.nan_to_num(trace_A_batch, nan=0.0)
    
    if np.isnan(trace_B_batch).any():
        logger.warning(f"NaNs detected in Run B (lambda={lambda_B}) trace. Imputing with 0.0.")
        trace_B_batch = np.nan_to_num(trace_B_batch, nan=0.0)

    mean_trace_A = np.mean(trace_A_batch, axis=0)
    std_trace_A = np.std(trace_A_batch, axis=0)

    mean_trace_B = np.mean(trace_B_batch, axis=0)
    std_trace_B = np.std(trace_B_batch, axis=0)

    return trace_A_batch, trace_B_batch, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B


def plot_results(traj_A, traj_B, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B, config):
    logger.info("Generating Figure 4 Beacon Plot.")
    output_plot = config["paths"]["output_plot"]
    lambda_A = config["intervention"]["lambda_A"]
    lambda_B = config["intervention"]["lambda_B"]
    num_runs = config["experiment"]["num_runs"]

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig = plt.figure(figsize=(18, 6))

    # Panel A: The Pathology
    ax1 = fig.add_subplot(131, projection="3d")
    tA_np = np.array(traj_A)
    ax1.plot(tA_np[:, 0], tA_np[:, 1], tA_np[:, 2], color="red", alpha=0.7, linewidth=1)
    ax1.scatter(tA_np[0, 0], tA_np[0, 1], tA_np[0, 2], color="black", s=50, label="x0 (Synthetically Degraded)")
    ax1.set_title(f"Panel A: Degraded Pathology (λ={lambda_A})")
    ax1.legend()

    # Panel B: The Phase Space Rescue
    ax2 = fig.add_subplot(132, projection="3d")
    tB_np = np.array(traj_B)
    ax2.plot(tB_np[:, 0], tB_np[:, 1], tB_np[:, 2], color="green", alpha=0.7, linewidth=1)
    ax2.scatter(tB_np[0, 0], tB_np[0, 1], tB_np[0, 2], color="black", s=50, label="x0 (Synthetically Degraded)")
    ax2.set_title(f"Panel B: Therapeutic Rescue (λ={lambda_B})")
    ax2.legend()

    # Panel C: Thermodynamic Restoration
    ax3 = fig.add_subplot(133)
    steps = np.arange(len(mean_trace_A))

    ax3.plot(
        steps, mean_trace_A, color="red", label=f"Run A: Degraded (λ={lambda_A})", linestyle="--"
    )
    ax3.fill_between(
        steps, mean_trace_A - std_trace_A, mean_trace_A + std_trace_A, color="red", alpha=0.2
    )

    ax3.plot(steps, mean_trace_B, color="green", label=f"Run B: Rescued (λ={lambda_B})")
    ax3.fill_between(
        steps, mean_trace_B - std_trace_B, mean_trace_B + std_trace_B, color="green", alpha=0.2
    )

    ax3.set_xlabel("Simulation Steps")
    ax3.set_ylabel("Effective Hessian Trace")
    ax3.set_title(f"Panel C: Thermodynamic Restoration (n={num_runs})")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    logger.info(f"Figure 4 successfully generated and saved to: {output_plot}")


def save_results(
    traj_A_batch, traj_B_batch, trace_A_batch, trace_B_batch, mean_A, std_A, mean_B, std_B, config
):
    output_metrics = config["paths"]["output_metrics"]
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)

    tA_clean = np.array(trace_A_batch).flatten()
    tB_clean = np.array(trace_B_batch).flatten()

    ks_stat, ks_pval = ks_2samp(tA_clean, tB_clean)
    wd = wasserstein_distance(tA_clean, tB_clean)
    d = pg.compute_effsize(tA_clean, tB_clean, eftype="cohen")

    data = {
        "config": config,
        "RunA": {
            "lambda": config["intervention"]["lambda_A"],
            "mean_trace": np.array(mean_A).tolist(),
            "std_trace": np.array(std_A).tolist(),
            "overall_mean": float(np.mean(tA_clean)),
            "overall_std": float(np.std(tA_clean)),
        },
        "RunB": {
            "lambda": config["intervention"]["lambda_B"],
            "mean_trace": np.array(mean_B).tolist(),
            "std_trace": np.array(std_B).tolist(),
            "overall_mean": float(np.mean(tB_clean)),
            "overall_std": float(np.std(tB_clean)),
        },
        "Comparisons": {
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_pval),
            "wasserstein_distance": float(wd),
            "cohens_d": float(d),
        },
    }

    with open(output_metrics, "w") as f:
        json.dump(data, f, indent=2)

    wandb.log(data["Comparisons"])
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

    wandb.init(project="worm_gait", name="07_worm_gait_intervention", config=config)
    
    # Establish lineage
    weights_path = config["paths"]["model_weights"]
    wandb.run.use_artifact("05_worm_gait_decline_trained_engine:latest", type="model")

    graph, x0, key = setup_experiment(config)

    traj_A_batch, traj_B_batch = run_experiment(graph, x0, key, config)

    trace_A_batch, trace_B_batch, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B = (
        calculate_metrics(graph, traj_A_batch, traj_B_batch, config)
    )

    plot_results(
        traj_A_batch[0],
        traj_B_batch[0],
        mean_trace_A,
        std_trace_A,
        mean_trace_B,
        std_trace_B,
        config,
    )

    save_results(
        traj_A_batch,
        traj_B_batch,
        trace_A_batch,
        trace_B_batch,
        mean_trace_A,
        std_trace_A,
        mean_trace_B,
        std_trace_B,
        config,
    )
    wandb.finish()

if __name__ == "__main__":
    main()
