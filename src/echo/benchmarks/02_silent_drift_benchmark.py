import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from src.echo.architecture.observer import MarkovBlanketObserver

def main():
    print("Running Silent Drift Benchmark...")
    
    d_internal = 16
    d_sensory = 8
    d_active = 16
    d_external = 16
    d_state = d_internal + d_sensory + d_active + d_external

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)

    # 1. Setup
    observer_healthy = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=32,
        ebm_depth=2,
        n_steps=1,
        temperature=1.0,
        key=k1,
        D_s=None
    )

    D_s_zero = jnp.zeros((d_sensory, d_sensory))
    
    # We must replace D_s in the top-level hull, as well as inside the unrollers' flow_factors.
    observer_blind = eqx.tree_at(
        lambda tree: (
            tree.hull.D_s,
            tree.thermalizer.graph.sites[0].factor.base.hull.D_s,
            tree.forced_thermalizer.flow_factor.hull.D_s
        ),
        observer_healthy,
        (D_s_zero, D_s_zero, D_s_zero),
        is_leaf=lambda x: x is None
    )

    # 2. Simulation
    seq_len = 100
    dt = 0.01
    x_init = jnp.zeros(d_state)

    # Linear drift added to the sensory slice of the state over time
    drift = jnp.linspace(0.0, 5.0, seq_len)
    data_seq = jnp.zeros((seq_len, d_sensory))
    for i in range(d_sensory):
        data_seq = data_seq.at[:, i].set(drift)

    # Unroll both observers
    traj_healthy = observer_healthy.forced_unroll(k2, x_init, dt, data_seq)
    traj_blind = observer_blind.forced_unroll(k2, x_init, dt, data_seq)

    # 3. Metrics to Extract over time
    div_healthy = jax.vmap(lambda x: jnp.linalg.norm(x - x_init))(traj_healthy)
    div_blind = jax.vmap(lambda x: jnp.linalg.norm(x - x_init))(traj_blind)

    def compute_surprisal(obs, x):
        def energy_fn(state):
            state_obs = obs.hull.apply_sensory_degradation(state)
            e, _ = obs.ebm(state_obs)
            return e
        grad_e = jax.grad(energy_fn)(x)
        return jnp.linalg.norm(obs.hull.partition(grad_e)["sensory"])

    surp_healthy = jax.vmap(lambda x: compute_surprisal(observer_healthy, x))(traj_healthy)
    surp_blind = jax.vmap(lambda x: compute_surprisal(observer_blind, x))(traj_blind)

    # Convert to numpy for plotting
    div_healthy_np = np.array(div_healthy)
    div_blind_np = np.array(div_blind)
    surp_healthy_np = np.array(surp_healthy)
    surp_blind_np = np.array(surp_blind)

    # 4. Output
    os.makedirs("output/echo", exist_ok=True)
    plot_path = "output/echo/silent_drift.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Subplot 1: Physical State Divergence
    ax1.plot(div_healthy_np, label="Healthy (D_s = None)", color="blue")
    ax1.plot(div_blind_np, label="Blind (D_s = 0)", color="red", linestyle="--")
    ax1.set_title("Physical State Divergence over time")
    ax1.set_ylabel("L2 Norm (x_current - x_init)")
    ax1.legend()
    
    # Subplot 2: Internal Surprisal
    ax2.plot(surp_healthy_np, label="Healthy (D_s = None)", color="blue")
    ax2.plot(surp_blind_np, label="Blind (D_s = 0)", color="red", linestyle="--")
    ax2.set_title("Internal Surprisal (Gradient Magnitude)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("L2 Norm of grad(E)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Diagnosis Readout
    mean_div_h = np.mean(div_healthy_np)
    mean_div_b = np.mean(div_blind_np)
    mean_surp_h = np.mean(surp_healthy_np)
    mean_surp_b = np.mean(surp_blind_np)

    print("--- Results Diagnosis ---")
    print(f"Healthy Divergence (mean): {mean_div_h:.4f}")
    print(f"Blind Divergence (mean): {mean_div_b:.4f}")
    print(f"Healthy Surprisal (mean): {mean_surp_h:.4f}")
    print(f"Blind Surprisal (mean): {mean_surp_b:.4f}")
    
    # Blind system should have exactly 0 surprisal on the sensory partition.
    if np.isclose(mean_surp_b, 0.0, atol=1e-5) and mean_surp_h > 0.0:
        print("DIAGNOSIS SUCCESS: The blind system experienced exactly zero surprisal. Silent drift confirmed.")
    else:
        print("DIAGNOSIS FAILED: The blind system generated surprisal, meaning it was not truly blind.")

if __name__ == "__main__":
    main()
