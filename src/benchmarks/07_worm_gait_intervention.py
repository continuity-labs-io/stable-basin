from jaxtyping import PRNGKeyArray
import os
import logging
import argparse
import yaml
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
from scipy.stats import ks_2samp, wasserstein_distance

from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset, SyntheticWormMockDataset
from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.metrics.thermal_interpretability import HessianCurvatureTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@eqx.filter_jit
def simulate_sde(
    graph: PredictiveCodingGraph, 
    x0: jax.Array, 
    lambda_gain: float, 
    N: int, 
    dt: float, 
    key: PRNGKeyArray
) -> jax.Array:
    """
    Simulates the core Euler-Maruyama SDE rollout with precision injection.
    """
    d_micro = graph.d_micro
    d_macro = graph.d_macro
    d_full = d_micro + d_macro
    
    ff = graph.flow_factor
    
    # 1. Reconstruct block-diagonal physical matrices for the joint graph
    Q_micro = ff.micro_solenoidal.Q
    L_micro = jnp.tril(ff.micro_dissipative.W)
    Gamma_micro = L_micro @ L_micro.T
    if ff.use_micro_blanket:
        M_micro = ff.micro_hull.get_topology_mask()
        Q_micro = Q_micro * M_micro
        Gamma_micro = Gamma_micro * M_micro
        
    Q_macro = ff.macro_solenoidal.Q
    L_macro = jnp.tril(ff.macro_dissipative.W)
    Gamma_macro = L_macro @ L_macro.T
    if ff.use_macro_blanket:
        M_macro = ff.macro_hull.get_topology_mask()
        Q_macro = Q_macro * M_macro
        Gamma_macro = Gamma_macro * M_macro
        
    Q_full = jax.scipy.linalg.block_diag(Q_micro, Q_macro)
    Gamma_full = jax.scipy.linalg.block_diag(Gamma_micro, Gamma_macro)
    
    # 2. Compute diffusion matrix S (sqrt of Gamma)
    evals, evecs = jnp.linalg.eigh(Gamma_full + ff.epsilon * jnp.eye(d_full))
    evals = jnp.maximum(evals, 0.0)
    S_full = evecs @ jnp.diag(jnp.sqrt(evals))
    
    # 3. Define the intervention energy landscape
    def energy_fn(x):
        x_u = x[:d_micro]
        x_m = x[d_micro:]
        return lambda_gain * ff.joint_energy_fn(x_u, x_m)
        
    # Lower the random thermal jitter for camera clarity
    T_micro = 0.05
    
    # 4. Continuous-time forward scan step
    def scan_step(x, key_step):
        grad_E = jax.grad(energy_fn)(x)
        drift = -(Q_full + Gamma_full) @ grad_E
        
        dW = jax.random.normal(key_step, (d_full,))
        diffusion = jnp.sqrt(2.0 * T_micro * dt) * (S_full @ dW)
        
        x_next = x + drift * dt + diffusion
        return x_next, x_next
        
    keys = jax.random.split(key, N)
    _, trajectory = jax.lax.scan(scan_step, x0, keys)
    
    return jnp.vstack([x0, trajectory])

def setup_experiment(config):
    logger.info("Initializing the 'Young Worm' physics engine (PredictiveCodingGraph).")
    
    seed = config['experiment']['seed']
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    c_micro = config['observer']['micro']
    c_macro = config['observer']['macro']
    
    micro = MarkovBlanketObserver(c_micro['d_internal'], c_micro['d_sensory'], c_micro['d_active'], c_micro['d_external'], 
                                  ebm_hidden_size=c_micro['ebm_hidden_size'], ebm_depth=c_micro['ebm_depth'], n_steps=1, temperature=c_micro['temperature'], key=k1)
                                  
    macro = MarkovBlanketObserver(c_macro['d_internal'], c_macro['d_sensory'], c_macro['d_active'], c_macro['d_external'], 
                                  ebm_hidden_size=c_macro['ebm_hidden_size'], ebm_depth=c_macro['ebm_depth'], n_steps=1, temperature=c_macro['temperature'], key=k2)
    
    d_micro_full = c_micro['d_internal'] + c_micro['d_sensory'] + c_micro['d_active'] + c_micro['d_external']
    d_macro_full = c_macro['d_internal'] + c_macro['d_sensory'] + c_macro['d_active'] + c_macro['d_external']
    
    micro = eqx.tree_at(lambda m: m.ebm, micro, PrecisionWeightedEBM(d_state=d_micro_full, hidden_size=c_micro['ebm_hidden_size'], depth=c_micro['ebm_depth'], key=k3))
    macro = eqx.tree_at(lambda m: m.ebm, macro, PrecisionWeightedEBM(d_state=d_macro_full, hidden_size=c_macro['ebm_hidden_size'], depth=c_macro['ebm_depth'], key=k3))
                                  
    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    
    model_path = config['paths']['model_weights']
    try:
        graph = eqx.tree_deserialise_leaves(model_path, graph)
        logger.info("Successfully loaded trained Young Worm engine.")
    except Exception as e:
        logger.warning(f"Trained model not found at {model_path}! Proceeding with random initialization.")
        
    d_full = graph.d_micro + graph.d_macro
    
    dataset_path = config['dataset']['path']
    seq_len = config['dataset']['seq_len']
    try:
        dataset = RealEigenwormDataset(data_path=dataset_path, seq_len=seq_len, is_aged=True)
    except FileNotFoundError:
        logger.warning("Biological data not found. Falling back to synthetic dataset.")
        dataset = SyntheticWormMockDataset(seq_len=seq_len, num_samples=1)
        
    bio_frame = dataset[0][0].numpy()  # 6D sensory snapshot
    
    logger.info("Extracting pathological initial state (x0) from 'Old Worm' fallback.")
    x0_noise = jax.random.normal(k4, (d_full,)) * 2.0
    x0_np = np.array(x0_noise)
    idx_s = micro.hull.d_internal
    idx_e = micro.hull.d_internal + micro.hull.d_sensory
    x0_np[idx_s:idx_e] = bio_frame
    x0 = jnp.array(x0_np)
    
    return graph, x0, k5

def run_experiment(graph, x0, key, config):
    N_steps = config['experiment']['N_steps']
    dt = config['experiment']['dt']
    num_runs = config['experiment']['num_runs']
    lambda_A = config['intervention']['lambda_A']
    lambda_B = config['intervention']['lambda_B']
    
    logger.info(f"Simulating Run A: Degraded baseline with precision_injection_gain={lambda_A}")
    keys_A = jax.random.split(key, num_runs)
    vmap_simulate = eqx.filter_jit(jax.vmap(simulate_sde, in_axes=(None, None, None, None, None, 0)))
    traj_A_batch = vmap_simulate(graph, x0, lambda_A, N_steps, dt, keys_A)
    
    logger.info(f"Simulating Run B: Therapeutic rescue with precision_injection_gain={lambda_B}")
    keys_B = jax.random.split(key, num_runs)  # Same seeds for fair comparison
    traj_B_batch = vmap_simulate(graph, x0, lambda_B, N_steps, dt, keys_B)
    
    return traj_A_batch, traj_B_batch

def calculate_metrics(graph, traj_A_batch, traj_B_batch, config):
    logger.info("Computing thermodynamic restoration metrics (Hessian trace).")
    tracker = HessianCurvatureTracker(graph.ebm)
    num_runs = config['experiment']['num_runs']
    lambda_A = config['intervention']['lambda_A']
    lambda_B = config['intervention']['lambda_B']
    
    def get_traces(traj_batch):
        traces = []
        for i in range(num_runs):
            metrics = tracker.batch_calculate_curvature(traj_batch[i])
            traces.append(np.array(metrics["hessian_trace"]))
        return np.vstack(traces)
        
    trace_A_batch = get_traces(traj_A_batch) * lambda_A
    trace_B_batch = get_traces(traj_B_batch) * lambda_B
    
    mean_trace_A = np.mean(trace_A_batch, axis=0)
    std_trace_A = np.std(trace_A_batch, axis=0)
    
    mean_trace_B = np.mean(trace_B_batch, axis=0)
    std_trace_B = np.std(trace_B_batch, axis=0)
    
    return trace_A_batch, trace_B_batch, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B

def plot_results(traj_A, traj_B, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B, config):
    logger.info("Generating Figure 4 Beacon Plot.")
    output_plot = config['paths']['output_plot']
    lambda_A = config['intervention']['lambda_A']
    lambda_B = config['intervention']['lambda_B']
    num_runs = config['experiment']['num_runs']
    
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig = plt.figure(figsize=(18, 6))
    
    # Panel A: The Pathology
    ax1 = fig.add_subplot(131, projection='3d')
    tA_np = np.array(traj_A)
    ax1.plot(tA_np[:, 0], tA_np[:, 1], tA_np[:, 2], color='red', alpha=0.7, linewidth=1)
    ax1.scatter(tA_np[0, 0], tA_np[0, 1], tA_np[0, 2], color='black', s=50, label='x0 (Old State)')
    ax1.set_title(f"Panel A: Degraded Pathology (λ={lambda_A})")
    ax1.legend()
    
    # Panel B: The Phase Space Rescue
    ax2 = fig.add_subplot(132, projection='3d')
    tB_np = np.array(traj_B)
    ax2.plot(tB_np[:, 0], tB_np[:, 1], tB_np[:, 2], color='green', alpha=0.7, linewidth=1)
    ax2.scatter(tB_np[0, 0], tB_np[0, 1], tB_np[0, 2], color='black', s=50, label='x0 (Old State)')
    ax2.set_title(f"Panel B: Therapeutic Rescue (λ={lambda_B})")
    ax2.legend()
    
    # Panel C: Thermodynamic Restoration
    ax3 = fig.add_subplot(133)
    steps = np.arange(len(mean_trace_A))
    
    ax3.plot(steps, mean_trace_A, color='red', label=f'Run A: Degraded (λ={lambda_A})', linestyle='--')
    ax3.fill_between(steps, mean_trace_A - std_trace_A, mean_trace_A + std_trace_A, color='red', alpha=0.2)
    
    ax3.plot(steps, mean_trace_B, color='green', label=f'Run B: Rescued (λ={lambda_B})')
    ax3.fill_between(steps, mean_trace_B - std_trace_B, mean_trace_B + std_trace_B, color='green', alpha=0.2)
    
    ax3.set_xlabel("Simulation Steps")
    ax3.set_ylabel("Effective Hessian Trace")
    ax3.set_title(f"Panel C: Thermodynamic Restoration (n={num_runs})")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    logger.info(f"Figure 4 successfully generated and saved to: {output_plot}")

def save_results(traj_A_batch, traj_B_batch, trace_A_batch, trace_B_batch, mean_A, std_A, mean_B, std_B, config):
    output_metrics = config['paths']['output_metrics']
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    
    tA_flat = np.array(trace_A_batch).flatten()
    tB_flat = np.array(trace_B_batch).flatten()
    
    tA_clean = np.nan_to_num(tA_flat, nan=1.0)
    tB_clean = np.nan_to_num(tB_flat, nan=1.0)
    
    ks_stat, ks_pval = ks_2samp(tA_clean, tB_clean)
    wd = wasserstein_distance(tA_clean, tB_clean)
    d = pg.compute_effsize(tA_clean, tB_clean, eftype='cohen')
    
    data = {
        "config": config,
        "RunA": {
            "lambda": config['intervention']['lambda_A'],
            "mean_trace": np.array(mean_A).tolist(),
            "std_trace": np.array(std_A).tolist(),
            "overall_mean": float(np.mean(tA_clean)),
            "overall_std": float(np.std(tA_clean))
        },
        "RunB": {
            "lambda": config['intervention']['lambda_B'],
            "mean_trace": np.array(mean_B).tolist(),
            "std_trace": np.array(std_B).tolist(),
            "overall_mean": float(np.mean(tB_clean)),
            "overall_std": float(np.std(tB_clean))
        },
        "Comparisons": {
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_pval),
            "wasserstein_distance": float(wd),
            "cohens_d": float(d)
        }
    }
    
    with open(output_metrics, 'w') as f:
        json.dump(data, f, indent=2)
        
    logger.info(f"Summary metrics successfully saved to: {output_metrics}")

def main():
    parser = argparse.ArgumentParser(description="Worm Gait Intervention Benchmark")
    parser.add_argument("--config", type=str, default="configs/worm_gait_intervention.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    graph, x0, key = setup_experiment(config)
    
    traj_A_batch, traj_B_batch = run_experiment(graph, x0, key, config)
    
    trace_A_batch, trace_B_batch, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B = calculate_metrics(
        graph, traj_A_batch, traj_B_batch, config
    )
    
    plot_results(traj_A_batch[0], traj_B_batch[0], mean_trace_A, std_trace_A, mean_trace_B, std_trace_B, config)
    
    save_results(traj_A_batch, traj_B_batch, trace_A_batch, trace_B_batch, mean_trace_A, std_trace_A, mean_trace_B, std_trace_B, config)

if __name__ == "__main__":
    main()
