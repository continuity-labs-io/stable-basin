import os
import json
import logging
import wandb
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Worm Gait Pharmacological Translation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/worm_gait_experiments.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load JSON Data
    sweep_metrics_path = "output/benchmarks/worm_gait/08_lambda_sweep_metrics.json"
    if not os.path.exists(sweep_metrics_path):
        logger.error(f"Sweep metrics file not found: {sweep_metrics_path}")
        return
        
    with open(sweep_metrics_path, "r") as f:
        sweep_data = json.load(f)
        
    lambdas = []
    R_values = []
    for k, v in sweep_data.items():
        lambdas.append(float(k))
        R_values.append(v["R_lambda"])
        
    lambdas = np.array(lambdas)
    R_values = np.array(R_values)
    
    # 2. Find Optimal Dose
    logger.info("Finding optimal dose corresponding to maximum therapeutic rescue...")
    max_idx = np.argmax(R_values)
    optimal_lambda = lambdas[max_idx]
    max_rescue_r = R_values[max_idx]
    logger.info(f"Optimal Lambda: {optimal_lambda:.4f}, Max Rescue R: {max_rescue_r:.4f}")

    wandb.init(project="worm_gait", name="09_pharmacological_translation", config=config)

    # 3. Serialization
    output_metrics = "output/benchmarks/worm_gait/09_clinical_translation_metrics.json"
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    metrics_data = {
        "Optimal_Lambda": float(optimal_lambda),
        "Max_Rescue_R": float(max_rescue_r)
    }
    with open(output_metrics, "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    wandb.log(metrics_data)
    logger.info(f"Metrics saved to {output_metrics}")

    # 4. Visualization
    output_plot = "output/benchmarks/worm_gait/09_pharmacological_curve.png"
    plt.figure(figsize=(10, 6))
    
    # Plot raw points
    plt.scatter(lambdas, R_values, color='blue', label=r'Measured $R(\lambda)$', zorder=5)
    
    # Add vertical dashed line for Optimal Lambda
    plt.axvline(x=optimal_lambda, color='green', linestyle='--', label=f'Optimal Dose = {optimal_lambda:.3f}')
    
    # Add text box
    textstr = f'Optimal Dose: {optimal_lambda:.3f}\nMax R: {max_rescue_r:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.xscale('log')
    plt.xlabel(r"Inverse-Temperature Scaling ($\lambda$)")
    plt.ylabel(r"Therapeutic Rescue $R(\lambda)$")
    plt.title("Pharmacological Translation & Dose-Response Curve")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(output_plot, dpi=300)
    wandb.log({"09_pharmacological_curve": wandb.Image(output_plot)})
    plt.close()
    
    artifact = wandb.Artifact("09_clinical_translation_metrics", type="metrics")
    artifact.add_file(output_metrics)
    wandb.log_artifact(artifact)
    wandb.finish()
    logger.info(f"Curve plot saved to {output_plot} and logged to wandb")

if __name__ == "__main__":
    main()
