import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.metrics.thermal_interpretability import HessianCurvatureTracker

def main():
    print("Running Concurrent Contention Benchmark...")
    
    # Setup
    d_internal = 8
    d_sensory = 8
    d_active = 8
    d_external = 8
    d_state = d_internal + d_sensory + d_active + d_external
    
    key = jax.random.PRNGKey(42)
    obs_key, sim_key = jax.random.split(key, 2)
    
    observer = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=32,
        ebm_depth=2,
        n_steps=1,
        temperature=1.0,
        key=obs_key
    )
    
    # Boost friction
    observer = eqx.tree_at(
        lambda o: o.dissipative.W, 
        observer, 
        observer.dissipative.W * 5.0
    )
    
    # Simulation Params
    seq_len = 200
    dt = 0.01
    x_init = jnp.zeros(d_state)
    
    # Phase A: Stressor 1 (Sensory)
    omega_1 = jnp.zeros((seq_len, d_state))
    # Sensory slice is from d_internal to d_internal + d_sensory
    s_start = d_internal
    s_end = d_internal + d_sensory
    omega_1 = omega_1.at[:, s_start:s_end].set(10.0)
    
    # Phase B: Stressor 2 (Active)
    omega_2 = jnp.zeros((seq_len, d_state))
    # Active slice is from d_internal + d_sensory to d_internal + d_sensory + d_active
    a_start = s_end
    a_end = a_start + d_active
    omega_2 = omega_2.at[:, a_start:a_end].set(10.0)
    
    # Phase C: Contention
    omega_3 = omega_1 + omega_2
    
    # Unroll
    key_A, key_B, key_C = jax.random.split(sim_key, 3)
    
    traj_A = observer.forced_unroll(key_A, x_init, dt, seq=None, omega_seq=omega_1)
    traj_B = observer.forced_unroll(key_B, x_init, dt, seq=None, omega_seq=omega_2)
    traj_C = observer.forced_unroll(key_C, x_init, dt, seq=None, omega_seq=omega_3)
    
    # Metrics
    def compute_divergence(traj):
        return jax.vmap(lambda x: jnp.linalg.norm(x - x_init))(traj)
        
    div_A = compute_divergence(traj_A)
    div_B = compute_divergence(traj_B)
    div_C = compute_divergence(traj_C)
    
    tracker = HessianCurvatureTracker(observer.ebm)
    
    metrics_seq_A = tracker.batch_calculate_curvature(traj_A)
    metrics_seq_B = tracker.batch_calculate_curvature(traj_B)
    metrics_seq_C = tracker.batch_calculate_curvature(traj_C)
    
    nullity_seq_A = metrics_seq_A["hessian_nullity"]
    nullity_seq_B = metrics_seq_B["hessian_nullity"]
    nullity_seq_C = metrics_seq_C["hessian_nullity"]
    
    final_A = traj_A[-1]
    final_B = traj_B[-1]
    final_C = traj_C[-1]
    
    metrics_A = tracker.calculate_curvature(final_A)
    metrics_B = tracker.calculate_curvature(final_B)
    metrics_C = tracker.calculate_curvature(final_C)
    
    trace_A = metrics_A["hessian_trace"]
    trace_B = metrics_B["hessian_trace"]
    trace_C = metrics_C["hessian_trace"]
    
    nullity_A = metrics_A["hessian_nullity"]
    nullity_B = metrics_B["hessian_nullity"]
    nullity_C = metrics_C["hessian_nullity"]
    
    rank_A = metrics_A["hessian_rank"]
    rank_B = metrics_B["hessian_rank"]
    rank_C = metrics_C["hessian_rank"]
    
    # Plotting
    os.makedirs("output/echo", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top Plot: Divergence
    ax1.plot(div_A, label="Phase A (Sensory Stressor)", color="blue")
    ax1.plot(div_B, label="Phase B (Active Stressor)", color="green")
    ax1.plot(div_C, label="Phase C (Contention)", color="red")
    ax1.set_title("Physical State Divergence under Contention")
    ax1.set_ylabel("L2 Divergence from Origin")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom Plot: Route Diversity (Hessian Nullity)
    ax2.plot(nullity_seq_A, label="Phase A Nullity", color="blue", linestyle="--")
    ax2.plot(nullity_seq_B, label="Phase B Nullity", color="green", linestyle="--")
    ax2.plot(nullity_seq_C, label="Phase C Nullity", color="red", linestyle="--")
    ax2.set_title("Route Diversity Collapse (Hessian Nullity)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Degenerate Dimensions (Nullity)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("output/echo/contention_benchmark.png")
    plt.close()
    
    print("Plot saved to output/echo/contention_benchmark.png")
    print("--- Final Results ---")
    print(f"Phase A | Divergence: {div_A[-1]:.4f} | Hessian Trace: {trace_A:.4f} | Rank: {rank_A} | Nullity: {nullity_A}")
    print(f"Phase B | Divergence: {div_B[-1]:.4f} | Hessian Trace: {trace_B:.4f} | Rank: {rank_B} | Nullity: {nullity_B}")
    print(f"Phase C | Divergence: {div_C[-1]:.4f} | Hessian Trace: {trace_C:.4f} | Rank: {rank_C} | Nullity: {nullity_C}")
    print("---------------------")

if __name__ == "__main__":
    main()
