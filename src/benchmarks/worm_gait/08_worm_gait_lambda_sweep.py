import os
import argparse
import yaml
import json
import logging
import wandb
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

from src.benchmarks.worm_gait.core import setup_experiment, simulate_sde
from src.echo.metrics.energy_landscape import batch_calculate_curvature

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_traces(graph, traj_batch, num_runs):
    energy_fn = lambda x: graph.ebm(x)[0]
    traces = []
    for i in range(num_runs):
        metrics = batch_calculate_curvature(energy_fn, traj_batch[i])
        traces.append(np.array(metrics["hessian_trace"]))
    return np.vstack(traces)

def main():
    parser = argparse.ArgumentParser(description="Worm Gait Lambda Sweep Benchmark")
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

    wandb.init(project="worm_gait", name="08_worm_gait_lambda_sweep", config=config)
    wandb.run.use_artifact("05_worm_gait_decline_trained_engine:latest", type="model")

    graph, x0, key = setup_experiment(config)

    lambdas = config["intervention"]["lambda_sweep"]
    lambda_baseline = config["intervention"]["lambda_A"]

    N_steps = config["experiment"]["N_steps"]
    dt = config["experiment"]["dt"]
    num_runs = config["experiment"]["num_runs"]

    vmap_simulate = eqx.filter_jit(
        jax.vmap(simulate_sde, in_axes=(None, None, None, None, None, 0))
    )

    results = {}
    mean_traces_per_lambda = []
    lambdas_for_plot = []

    # Calculate baseline
    logger.info(f"Simulating Baseline: lambda={lambda_baseline}")
    keys_baseline = jax.random.split(key, num_runs)
    traj_baseline_batch = vmap_simulate(graph, x0, lambda_baseline, N_steps, dt, keys_baseline)
    trace_baseline_batch = get_traces(graph, traj_baseline_batch, num_runs)
    
    if np.isnan(trace_baseline_batch).any():
        logger.warning(f"NaNs detected in Baseline (lambda={lambda_baseline}) trace. Imputing with 0.0.")
        trace_baseline_batch = np.nan_to_num(trace_baseline_batch, nan=0.0)
        
    baseline_flat = np.array(trace_baseline_batch).flatten()
    baseline_mean_trace = np.mean(trace_baseline_batch, axis=0)
    baseline_overall_mean = float(np.mean(baseline_flat))

    for lam in lambdas:
        logger.info(f"Simulating Sweep: lambda={lam}")
        # Use same keys for fair comparison
        traj_lam_batch = vmap_simulate(graph, x0, lam, N_steps, dt, keys_baseline)
        trace_lam_batch = get_traces(graph, traj_lam_batch, num_runs)
        
        if np.isnan(trace_lam_batch).any():
            logger.warning(f"NaNs detected in Sweep (lambda={lam}) trace. Imputing with 0.0.")
            trace_lam_batch = np.nan_to_num(trace_lam_batch, nan=0.0)
            
        lam_flat = np.array(trace_lam_batch).flatten()
        lam_mean_trace = np.mean(trace_lam_batch, axis=0)
        overall_lam_mean = float(np.mean(lam_flat))
        
        # Calculate Cohen's d between baseline flat trace and current lam flat trace
        # pingouin computes effect size based on two 1D arrays
        d = pg.compute_effsize(baseline_flat, lam_flat, eftype="cohen")
        
        results[float(lam)] = {
            "mean_trace": np.array(lam_mean_trace).tolist(),
            "mean_trace_overall": overall_lam_mean,
            "cohens_d": float(d)
        }
        mean_traces_per_lambda.append(overall_lam_mean)
        lambdas_for_plot.append(lam)

    output_metrics = "output/benchmarks/worm_gait/08_lambda_sweep_metrics.json"
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(results, f, indent=2)
    
    # Log numerical sweep results
    for lam, metric in results.items():
        wandb.log({"lambda": lam, "mean_trace_overall": metric["mean_trace_overall"], "cohens_d": metric["cohens_d"]})

    logger.info(f"Metrics saved to {output_metrics}")

    output_plot = "output/benchmarks/worm_gait/08_lambda_dose_response.png"
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.plot(lambdas_for_plot, mean_traces_per_lambda, marker='o', color='blue', label='Mean Effective Hessian Trace')
    plt.axhline(y=baseline_overall_mean, color='red', linestyle='--', label=f'Baseline (λ={lambda_baseline})')
    
    plt.xscale('log')
    plt.xticks(lambdas, labels=[str(l) for l in lambdas])
    
    plt.xlabel(r"Precision Injection Parameter ($\lambda$)")
    plt.ylabel("Mean Effective Hessian Trace")
    plt.title("Clinical Dose-Response Sweep")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(output_plot, dpi=300)
    wandb.log({"08_lambda_dose_response": wandb.Image(output_plot)})
    plt.close()
    
    artifact = wandb.Artifact("08_lambda_sweep_metrics", type="metrics")
    artifact.add_file(output_metrics)
    wandb.log_artifact(artifact)
    wandb.finish()
    logger.info(f"Dose-response plot saved to {output_plot} and logged to wandb")

if __name__ == "__main__":
    main()
