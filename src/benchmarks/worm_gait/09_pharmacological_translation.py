import os
import json
import logging
import wandb
import argparse
import yaml
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def hill_equation(x, bottom, top, ec50, hill_slope):
    """
    Computes the 4-parameter logistic (4PL) Hill equation for dose-response curves.

    Args:
        x: The concentration or dose (lambda parameter).
        bottom: The minimum asymptotic response value.
        top: The maximum asymptotic response value.
        ec50: The dose at which 50% of the maximum response is achieved.
        hill_slope: The slope factor characterizing the steepness of the curve.

    Returns:
        The calculated response value for the given dose.
    """
    return bottom + (top - bottom) / (1 + (ec50 / x)**hill_slope)

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
    
    # 2. Fit Sigmoid Curve (Hill Equation)
    logger.info("Fitting 4PL Hill equation to Dose-Response curve...")
    # Initial guesses: bottom = min R, top = max R, ec50 = median lambda, hill_slope = 1.0
    p0 = [np.min(R_values), np.max(R_values), np.median(lambdas), 1.0]
    
    # Bounds: bottom/top can be anything, ec50 bounded to strictly fall between min(lambdas) and max(lambdas)
    bounds = (
        [-np.inf, -np.inf, min(lambdas) + 1e-6, -np.inf], 
        [np.inf, np.inf, max(lambdas) - 1e-6, np.inf]
    )
    
    try:
        popt, pcov = curve_fit(hill_equation, lambdas, R_values, p0=p0, bounds=bounds, maxfev=10000)
        bottom, top, ec50, hill_slope = popt
        logger.info(f"Fitted EC50: {ec50:.4f}, Hill Slope: {hill_slope:.4f}")
    except RuntimeError as e:
        logger.error(f"Curve fitting failed: {e}")
        return

    wandb.init(project="worm_gait", name="09_pharmacological_translation", config=config)

    # 3. Serialization
    output_metrics = "output/benchmarks/worm_gait/09_clinical_translation_metrics.json"
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    metrics_data = {
        "EC50": float(ec50),
        "Hill_Slope": float(hill_slope),
        "Maximum_Asymptote": float(top),
        "Minimum_Asymptote": float(bottom)
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
    
    # Plot smooth fitted curve
    x_smooth = np.logspace(np.log10(min(lambdas)*0.5), np.log10(max(lambdas)*1.5), 200)
    y_smooth = hill_equation(x_smooth, *popt)
    plt.plot(x_smooth, y_smooth, color='red', label='4PL Hill Equation Fit', zorder=4)
    
    # Add vertical dashed line for EC50
    plt.axvline(x=ec50, color='green', linestyle='--', label=f'$EC_{{50}}$ = {ec50:.3f}')
    
    # Add text box
    textstr = f'$EC_{{50}}$: {ec50:.3f}'
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
