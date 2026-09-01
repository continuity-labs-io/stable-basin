"""
Hahne (2026) Decidability Diagnostic Benchmark

Reference: 
Hahne, 2026. "Beyond Return to Baseline: Reachability, Observability, and the Measurement of Physiological Margin"

This benchmark differentiates between two distinct failure modes in hierarchical Active Inference systems:
1. Policy Observability Failure (Sensory Blindness): The system loses access to sensory prediction errors,
   causing it to drift, but its internal structural integrity remains intact.
2. Reachability Collapse (Hardware/Structural Failure): The system's internal mechanical bonds (friction)
   are destroyed, permanently eliminating its physical attractor basins.

Mechanism:
- Phase 1 (Endogenous Failure): Both patients are subjected to a constant environmental drift (`omega_seq`)
  that pushes them away from the homeostatic origin (the minimum of the Waddington basin).
- Phase 2 (Intervention): A closed-loop proportional controller applies a restorative
  physical force. Crucially, this external device can ONLY stimulate the accessible exterior nodes 
  (the Markov Blanket: sensory and active states). It cannot reach into the internal hidden states.

Dynamics & Diagnosis:
- Patient A (Blindness) has broken sensory matrices (`D_s=0`) but an intact internal mechanical structure
  (strong friction `Gamma`). When the controller pulls the blanket into homeostasis, Patient A's healthy
  internal mechanical bonds physically drag the core internal states back to safety. Diagnosis: Rescuable.
- Patient B (Structural Collapse) has destroyed internal friction matrices (`Gamma -> 0`). When the 
  controller pulls the blanket into homeostasis, the internal states physically detach. Without friction
  to bind them to the blanket, they continue to drift irreversibly. Diagnosis: Irreversible Collapse.
"""

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from src.echo.architecture.observer import MarkovBlanketObserver

# Cite Hahne (2026) "Beyond Return to Baseline: Reachability, Observability, and
# the Measurement of Physiological Margin"

class QuadraticEBM(eqx.Module):
    d_state: int
    def __init__(self, d_state):
        self.d_state = d_state
    def __call__(self, x):
        return 0.5 * jnp.sum(x**2), jnp.eye(self.d_state)

def run_simulation():
    # Setup
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    d_internal = 4
    d_sensory = 4
    d_active = 4
    d_external = 4
    d_state = d_internal + d_sensory + d_active + d_external
    
    # Patient A (Blindness)
    patient_a = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=32,
        ebm_depth=1,
        n_steps=1,
        temperature=0.0,
        key=k1,
        D_s=jnp.zeros((d_sensory, d_sensory))  # Zeroed out sensors
    )
    
    # Patient B (Structural Collapse)
    patient_b = MarkovBlanketObserver(
        d_internal=d_internal,
        d_sensory=d_sensory,
        d_active=d_active,
        d_external=d_external,
        ebm_hidden_size=32,
        ebm_depth=1,
        n_steps=1,
        temperature=0.0,
        key=k2
    )
    
    # Inject QuadraticEBM and strong Brakes so that Patient A's healthy
    # components strongly restore
    strong_brakes = jnp.eye(d_state) * 5.0
    
    def apply_patient_mods(tree, brake_val):
        from src.echo.primitives.ebm import PrecisionWeightedEBM
        from src.echo.physics.dissipative import DissipativeFriction
        from src.echo.physics.solenoidal import SolenoidalFlow
        def _map_fn(x):
            if isinstance(x, DissipativeFriction):
                return eqx.tree_at(lambda p: p.W, x, brake_val)
            if isinstance(x, SolenoidalFlow):
                return eqx.tree_at(lambda p: p.W, x, jnp.zeros_like(x.W))
            if isinstance(x, PrecisionWeightedEBM):
                return QuadraticEBM(d_state)
            return x
        return jax.tree_util.tree_map(_map_fn, tree, is_leaf=lambda x: isinstance(x, (DissipativeFriction, SolenoidalFlow, PrecisionWeightedEBM)))

    patient_a = apply_patient_mods(patient_a, strong_brakes)
    patient_b = apply_patient_mods(patient_b, strong_brakes * 0.001)
    
    # The Simulation
    seq_len = 200
    dt = 0.01
    x_init = jnp.zeros(d_state)
    
    # Phase 1: Endogenous Failure Environmental drift omega_seq
    omega_seq = jnp.ones((seq_len, d_state)) * 0.5
    
    traj_a_p1 = patient_a.forced_unroll(k3, x_init, dt, seq=None, omega_seq=omega_seq)
    traj_b_p1 = patient_b.forced_unroll(k3, x_init, dt, seq=None, omega_seq=omega_seq)
    
    x_drifted_a = traj_a_p1[-1]
    x_drifted_b = traj_b_p1[-1]
    
    # Phase 2: The Intervention
    q_mask = jnp.zeros(d_state)
    idx_s = d_internal
    idx_a = idx_s + d_sensory
    idx_e = idx_a + d_active
    # Set to 1.0 for sensory and active
    q_mask = q_mask.at[idx_s:idx_e].set(1.0)
    
    # Unroll BOTH observers again starting from x_drifted to fight the omega_seq
    traj_a_p2 = patient_a.forced_unroll(k3, x_drifted_a, dt, seq=None, omega_seq=omega_seq, q_gain=10.0, q_mask=q_mask)
    traj_b_p2 = patient_b.forced_unroll(k3, x_drifted_b, dt, seq=None, omega_seq=omega_seq, q_gain=10.0, q_mask=q_mask)
    
    final_a = traj_a_p2[-1]
    final_b = traj_b_p2[-1]
    
    dist_a = jnp.linalg.norm(final_a)
    dist_b = jnp.linalg.norm(final_b)
    
    rescue_threshold = 1.0
    
    print(f"Patient A Diagnosis (dist: {dist_a}):")
    if dist_a < rescue_threshold:
        diag_a = "[DIAGNOSIS: POLICY_OBSERVABILITY_FAILURE] Hardware restored homeostasis. Physical Waddington basin is intact."
        print(diag_a)
    else:
        diag_a = "[DIAGNOSIS: REACHABILITY_COLLAPSE] Hardware failed. The physical attractor basin is irreversibly destroyed."
        print(diag_a)
        
    print(f"Patient B Diagnosis (dist: {dist_b}):")
    if dist_b < rescue_threshold:
        diag_b = "[DIAGNOSIS: POLICY_OBSERVABILITY_FAILURE] Hardware restored homeostasis. Physical Waddington basin is intact."
        print(diag_b)
    else:
        diag_b = "[DIAGNOSIS: REACHABILITY_COLLAPSE] Hardware failed. The physical attractor basin is irreversibly destroyed."
        print(diag_b)
        
    return traj_a_p1, traj_a_p2, traj_b_p1, traj_b_p2, diag_a, diag_b

def main():
    traj_a_p1, traj_a_p2, traj_b_p1, traj_b_p2, diag_a, diag_b = run_simulation()
    
    # Generate Matplotlib figure
    os.makedirs("output/echo", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    norm_a_p1 = jnp.linalg.norm(traj_a_p1, axis=-1)
    norm_a_p2 = jnp.linalg.norm(traj_a_p2, axis=-1)
    norm_a_full = jnp.concatenate([norm_a_p1, norm_a_p2])
    
    norm_b_p1 = jnp.linalg.norm(traj_b_p1, axis=-1)
    norm_b_p2 = jnp.linalg.norm(traj_b_p2, axis=-1)
    norm_b_full = jnp.concatenate([norm_b_p1, norm_b_p2])
    
    seq_len = len(norm_a_p1)
    time_steps = jnp.arange(len(norm_a_full))
    
    ax1.plot(time_steps, norm_a_full, label="L2 Norm")
    ax1.axvline(x=seq_len, color='r', linestyle='--', label="Hardware Restoration Triggered")
    
    diag_a_short = "POLICY_OBSERVABILITY_FAILURE" if "POLICY" in diag_a else "REACHABILITY_COLLAPSE"
    ax1.set_title(f"Patient A (Blindness)\n{diag_a_short}")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Physical State Divergence (L2 Norm)")
    ax1.legend()
    
    ax2.plot(time_steps, norm_b_full, label="L2 Norm")
    ax2.axvline(x=seq_len, color='r', linestyle='--', label="Hardware Restoration Triggered")
    
    diag_b_short = "POLICY_OBSERVABILITY_FAILURE" if "POLICY" in diag_b else "REACHABILITY_COLLAPSE"
    ax2.set_title(f"Patient B (Structural Collapse)\n{diag_b_short}")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Physical State Divergence (L2 Norm)")
    ax2.legend()
    
    plt.suptitle("Decidability Diagnostic\nRef: Hahne (2026) 'Beyond Return to Baseline: Reachability, Observability, and the Measurement of Physiological Margin'")
    plt.tight_layout()
    plt.savefig("output/echo/decidability_diagnostic.png")
    plt.close()

if __name__ == "__main__":
    main()
