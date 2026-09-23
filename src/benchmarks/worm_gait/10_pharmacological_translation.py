import os
import json
import logging
import argparse
import yaml
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx

import importlib
worm_gait_intervention = importlib.import_module("src.benchmarks.worm_gait.07_worm_gait_intervention")
setup_experiment = worm_gait_intervention.setup_experiment
simulate_sde = worm_gait_intervention.simulate_sde

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

def calculate_energies(graph, traj_batch):
    """
    Calculates the joint energy for every state in a batch of trajectories.

    Args:
        graph: The PredictiveCodingGraph containing the flow factor and partition sizes.
        traj_batch: A batch of simulated SDE trajectories.

    Returns:
        A numpy array containing the scalar joint energy for each state in the trajectories.
    """
    d_micro = graph.d_micro
    
    def joint_energy(x):
        x_micro = x[:d_micro]
        x_macro = x[d_micro:]
        return graph.flow_factor.joint_energy_fn(x_micro, x_macro)
    
    vmap_energy = jax.vmap(joint_energy)
    vmap_batch_energy = jax.vmap(vmap_energy)
    
    return np.array(vmap_batch_energy(traj_batch))

def main():
    parser = argparse.ArgumentParser(description="Worm Gait Pharmacological Translation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/worm_gait_intervention.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load JSON Data
    sweep_metrics_path = "output/echo/benchmarks/09_lambda_sweep_metrics.json"
    if not os.path.exists(sweep_metrics_path):
        logger.error(f"Sweep metrics file not found: {sweep_metrics_path}")
        return
        
    with open(sweep_metrics_path, "r") as f:
        sweep_data = json.load(f)
        
    lambdas = []
    traces = []
    for k, v in sweep_data.items():
        lambdas.append(float(k))
        traces.append(v["mean_trace_overall"])
        
    lambdas = np.array(lambdas)
    traces = np.array(traces)
    
    # 2. Fit Sigmoid Curve (Hill Equation)
    logger.info("Fitting 4PL Hill equation to Dose-Response curve...")
    # Initial guesses: bottom = min trace, top = max trace, ec50 = median lambda, hill_slope = 1.0
    p0 = [np.min(traces), np.max(traces), np.median(lambdas), 1.0]
    
    # Bounds: bottom/top can be anything, ec50 bounded to reasonable dose range, hill_slope can be anything
    bounds = (
        [-np.inf, -np.inf, min(lambdas)*0.1, -np.inf], 
        [np.inf, np.inf, max(lambdas)*10.0, np.inf]
    )
    
    try:
        popt, pcov = curve_fit(hill_equation, lambdas, traces, p0=p0, bounds=bounds, maxfev=10000)
        bottom, top, ec50, hill_slope = popt
        logger.info(f"Fitted EC50: {ec50:.4f}, Hill Slope: {hill_slope:.4f}")
    except RuntimeError as e:
        logger.error(f"Curve fitting failed: {e}")
        return

    # 3. Setup Physics Engine
    graph, x0, key = setup_experiment(config)

    N_steps = config["experiment"]["N_steps"]
    dt = config["experiment"]["dt"]
    num_runs = config["experiment"]["num_runs"]

    vmap_simulate = eqx.filter_jit(
        jax.vmap(simulate_sde, in_axes=(None, None, None, None, None, 0))
    )

    lambda_base = config["intervention"]["lambda_A"]
    lambda_rescue = float(ec50)
    
    # 4. The Trajectories
    logger.info(f"Simulating Pathological Baseline (lambda={lambda_base})")
    keys_base = jax.random.split(key, num_runs)
    traj_base = vmap_simulate(graph, x0, lambda_base, N_steps, dt, keys_base)
    
    logger.info(f"Simulating Therapeutic Rescue (lambda={lambda_rescue})")
    keys_rescue = jax.random.split(key, num_runs)
    traj_rescue = vmap_simulate(graph, x0, lambda_rescue, N_steps, dt, keys_rescue)

    # 5. Thermodynamic Translation (Delta G)
    logger.info("Calculating Thermodynamic Translation (Delta G)...")
    energy_base = calculate_energies(graph, traj_base)
    energy_rescue = calculate_energies(graph, traj_rescue)
    
    mean_energy_base = float(np.mean(energy_base))
    mean_energy_rescue = float(np.mean(energy_rescue))
    
    G_baseline = lambda_base * mean_energy_base
    G_rescue = lambda_rescue * mean_energy_rescue
    delta_G = abs(G_rescue - G_baseline)
    
    logger.info(f"G_baseline: {G_baseline:.4f}")
    logger.info(f"G_rescue: {G_rescue:.4f}")
    logger.info(f"Delta G: {delta_G:.4f}")

    # 6. Serialization
    output_metrics = "output/echo/benchmarks/10_clinical_translation_metrics.json"
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    metrics_data = {
        "EC50": float(ec50),
        "Hill_Slope": float(hill_slope),
        "Maximum_Asymptote": float(top),
        "Minimum_Asymptote": float(bottom),
        "G_baseline": G_baseline,
        "G_rescue": G_rescue,
        "Delta_G": delta_G
    }
    with open(output_metrics, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Metrics saved to {output_metrics}")

    # 7. Visualization
    output_plot = "output/echo/benchmarks/10_pharmacological_curve.png"
    plt.figure(figsize=(10, 6))
    
    # Plot raw points
    plt.scatter(lambdas, traces, color='blue', label='Measured Trace (SDE Rollout)', zorder=5)
    
    # Plot smooth fitted curve
    x_smooth = np.logspace(np.log10(min(lambdas)*0.5), np.log10(max(lambdas)*1.5), 200)
    y_smooth = hill_equation(x_smooth, *popt)
    plt.plot(x_smooth, y_smooth, color='red', label='4PL Hill Equation Fit', zorder=4)
    
    # Add vertical dashed line for EC50
    plt.axvline(x=ec50, color='green', linestyle='--', label=f'$EC_{{50}}$ = {ec50:.3f}')
    
    # Add text box with Delta G
    textstr = f'$EC_{{50}}$: {ec50:.3f}\n$\\Delta G$: {delta_G:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.xscale('log')
    plt.xlabel(r"Precision Injection Parameter ($\lambda$)")
    plt.ylabel("Mean Effective Hessian Trace")
    plt.title("Pharmacological Translation & Dose-Response Curve")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(output_plot, dpi=300)
    plt.close()
    logger.info(f"Curve plot saved to {output_plot}")

if __name__ == "__main__":
    main()
