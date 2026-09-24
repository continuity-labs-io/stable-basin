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
from scipy.stats import energy_distance

from src.benchmarks.worm_gait.core import setup_experiment, simulate_sde
from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

    d_internal = graph.hull.d_internal
    d_sensory = graph.hull.d_sensory

    def get_sensory_flat(traj_batch):
        sensory = traj_batch[:, :, d_internal : d_internal + d_sensory]
        return np.array(sensory).flatten()

    # Load clean biological data Y
    dataset_path = config["dataset"].get("intervention_path", "data/worm/EigenWorms_TEST.ts")
    try:
        ds_young = RealEigenwormDataset(data_path=dataset_path, seq_len=1000, inject_synthetic_degradation=False)
        import torch
        Y = torch.cat(ds_young.data).numpy().flatten()
    except FileNotFoundError:
        logger.warning(f"Biological data not found at {dataset_path}, falling back to synthetic mock data.")
        ds_young = SyntheticWormMockDataset(seq_len=1000, num_samples=5)
        Y = np.stack([ds_young[i][0].numpy() for i in range(5)]).flatten()


    results = {}
    R_per_lambda = []
    lambdas_for_plot = []

    # Calculate baseline
    logger.info(f"Simulating Baseline: lambda={lambda_baseline}")
    keys_baseline = jax.random.split(key, num_runs)
    traj_baseline_batch = vmap_simulate(graph, x0, lambda_baseline, N_steps, dt, keys_baseline)
    M_baseline = get_sensory_flat(traj_baseline_batch)
    
    if np.isnan(M_baseline).any():
        logger.warning(f"NaNs detected in Baseline (lambda={lambda_baseline}) trajectory. Imputing with 0.0.")
        M_baseline = np.nan_to_num(M_baseline, nan=0.0)
        
    dist_baseline = float(energy_distance(Y, M_baseline))

    for lam in lambdas:
        logger.info(f"Simulating Sweep: lambda={lam}")
        # Use same keys for fair comparison
        traj_lam_batch = vmap_simulate(graph, x0, lam, N_steps, dt, keys_baseline)
        M_lam = get_sensory_flat(traj_lam_batch)
        
        if np.isnan(M_lam).any():
            logger.warning(f"NaNs detected in Sweep (lambda={lam}) trajectory. Imputing with 0.0.")
            M_lam = np.nan_to_num(M_lam, nan=0.0)
            
        dist_lam = float(energy_distance(Y, M_lam))
        R_lam = 1.0 - (dist_lam / dist_baseline) if dist_baseline != 0 else 0.0
        
        results[float(lam)] = {
            "energy_distance": dist_lam,
            "R_lambda": R_lam
        }
        R_per_lambda.append(R_lam)
        lambdas_for_plot.append(lam)

    output_metrics = "output/benchmarks/worm_gait/08_lambda_sweep_metrics.json"
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(results, f, indent=2)
    
    # Log numerical sweep results
    for lam, metric in results.items():
        wandb.log({"lambda": lam, "energy_distance": metric["energy_distance"], "R_lambda": metric["R_lambda"]})

    logger.info(f"Metrics saved to {output_metrics}")

    output_plot = "output/benchmarks/worm_gait/08_lambda_dose_response.png"
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.plot(lambdas_for_plot, R_per_lambda, marker='o', color='blue', label=r'Therapeutic Rescue $R(\lambda)$')
    plt.axhline(y=0.0, color='red', linestyle='--', label=f'Baseline Rescue (λ={lambda_baseline})')
    
    plt.xscale('log')
    plt.xticks(lambdas, labels=[str(l) for l in lambdas])
    
    plt.xlabel(r"Inverse-Temperature Scaling ($\lambda$)")
    plt.ylabel(r"Therapeutic Rescue $R(\lambda)$")
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
