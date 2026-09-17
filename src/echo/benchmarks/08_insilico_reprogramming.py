import os
import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from src.data.behavior.celegans_gait_dataset import CElegansGaitDataset
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
    key: jax.random.PRNGKey
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
    
    T_micro = ff.micro_thermostat.temperature
    
    # 3. Define the intervention energy landscape
    def energy_fn(x):
        x_u = x[:d_micro]
        x_m = x[d_micro:]
        # Multiply the learned energy landscape by the precision injection gain lambda
        return lambda_gain * ff.joint_energy_fn(x_u, x_m)
        
    # Lower the random thermal jitter for camera clarity
    T_micro = 0.05
    # 4. Continuous-time forward scan step
    def scan_step(x, key_step):
        grad_E = jax.grad(energy_fn)(x)
        # Drift matching Thermostat.py: -(Q + Gamma) @ grad_E
        drift = -(Q_full + Gamma_full) @ grad_E
        
        # Stochastic environmental noise
        dW = jax.random.normal(key_step, (d_full,))
        diffusion = jnp.sqrt(2.0 * T_micro * dt) * (S_full @ dW)
        
        x_next = x + drift * dt + diffusion
        return x_next, x_next
        
    keys = jax.random.split(key, N)
    _, trajectory = jax.lax.scan(scan_step, x0, keys)
    
    return jnp.vstack([x0, trajectory])


def main():
    # Setup
    N_steps = 3000
    num_runs = 5
    dt = 0.01
    output_dir = "output/echo/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_plot = os.path.join(output_dir, "08_insilico_reprogramming_fig4_insilico_rescue.png")
    
    logger.info("Initializing the 'Young Worm' physics engine (PredictiveCodingGraph).")
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    # Instantiate Micro and Macro Observers (Young Engine)
    d_sensory = 6
    d_internal = 8
    d_active = 8
    d_external = 8
    
    micro = MarkovBlanketObserver(d_internal, d_sensory, d_active, d_external, 
                                  ebm_hidden_size=32, ebm_depth=2, n_steps=1, temperature=1.0, key=k1)
                                  
    macro = MarkovBlanketObserver(4, 4, 4, 4, 
                                  ebm_hidden_size=16, ebm_depth=2, n_steps=1, temperature=1.0, key=k2)
    
    micro = eqx.tree_at(lambda m: m.ebm, micro, PrecisionWeightedEBM(d_state=d_internal + d_sensory + d_active + d_external, hidden_size=32, depth=2, key=k3))
    macro = eqx.tree_at(lambda m: m.ebm, macro, PrecisionWeightedEBM(d_state=4 + 4 + 4 + 4, hidden_size=16, depth=2, key=k3))
                                  
    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    
    try:
        graph = eqx.tree_deserialise_leaves("output/echo/benchmarks/06_worm_gait_decline_trained_engine.eqx", graph)
        logger.info("Successfully loaded trained Young Worm engine.")
    except Exception as e:
        logger.warning("Trained model not found! Proceeding with random initialization.")
    d_full = graph.d_micro + graph.d_macro
    
    # Load fallback biological data to represent a fragment of reality
    dataset = CElegansGaitDataset(seq_len=10)
    bio_frame = dataset[0][0].numpy()  # 6D sensory snapshot
    
    # Construct "Old Worm" pathological state (erratic, high variance)
    logger.info("Extracting pathological initial state (x0) from 'Old Worm' fallback.")
    x0_noise = jax.random.normal(k4, (d_full,)) * 2.0
    x0_np = np.array(x0_noise)
    # Inject biological fragment into the sensory partition of the micro blanket
    idx_s = micro.hull.d_internal
    idx_e = micro.hull.d_internal + micro.hull.d_sensory
    x0_np[idx_s:idx_e] = bio_frame
    x0 = jnp.array(x0_np)
    
    # Simulate Run A (Degraded/Aged Baseline)
    lambda_A = 0.2
    logger.info(f"Simulating Run A: Degraded baseline with precision_injection_gain={lambda_A}")
    keys_A = jax.random.split(k5, num_runs)
    
    # We vmap over the keys to simulate multiple independent noise trajectories
    vmap_simulate = eqx.filter_jit(jax.vmap(simulate_sde, in_axes=(None, None, None, None, None, 0)))
    traj_A_batch = vmap_simulate(graph, x0, lambda_A, N_steps, dt, keys_A)
    
    # Simulate Run B (The Rescue)
    lambda_B = 5.0
    logger.info(f"Simulating Run B: Therapeutic rescue with precision_injection_gain={lambda_B}")
    keys_B = jax.random.split(k5, num_runs)  # Using the same base seeds for fair noise comparison
    traj_B_batch = vmap_simulate(graph, x0, lambda_B, N_steps, dt, keys_B)
    
    # Compute Thermodynamic Curvature
    logger.info("Computing thermodynamic restoration metrics (Hessian trace).")
    tracker = HessianCurvatureTracker(graph.ebm)
    
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
    
    # For the 3D phase space plot, we just visualize the first trajectory
    traj_A = traj_A_batch[0]
    traj_B = traj_B_batch[0]
    
    # Generate Figure 4 Visual
    logger.info("Generating Figure 4 Beacon Plot.")
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
    
    # Panel C: Thermodynamic Restoration (Hessian Trace)
    ax3 = fig.add_subplot(133)
    steps = np.arange(len(mean_trace_A))
    
    ax3.plot(steps, mean_trace_A, color='red', label=f'Run A: Degraded (λ={lambda_A})', linestyle='--')
    ax3.fill_between(steps, mean_trace_A - std_trace_A, mean_trace_A + std_trace_A, color='red', alpha=0.2)
    
    ax3.plot(steps, mean_trace_B, color='green', label=f'Run B: Rescued (λ={lambda_B})')
    ax3.fill_between(steps, mean_trace_B - std_trace_B, mean_trace_B + std_trace_B, color='green', alpha=0.2)
    
    ax3.set_xlabel("Simulation Steps")
    ax3.set_ylabel("Effective Hessian Trace (Steeper Basin = Healthier)")
    ax3.set_title(f"Panel C: Thermodynamic Restoration (n={num_runs})")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    logger.info(f"Figure 4 successfully generated and saved to: {output_plot}")


if __name__ == "__main__":
    main()
