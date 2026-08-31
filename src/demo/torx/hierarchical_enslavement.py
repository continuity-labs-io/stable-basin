"""
Phase 4: Torx Hierarchical Enslavement

Demonstrates the cybernetic loop: A slow, macroscopic order parameter projects
top-down precision to thermodynamically enslave fast, microscopic variables,
drastically reducing their entropy/variance via Free Energy minimization.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import logging
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torx import DFG, Site, ChainFactor
from torx.factor import AbstractReferenceFactor

from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TorxHierarchicalEnslavement")

class HierarchicalEnslavementFactor(AbstractReferenceFactor):
    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)
    
    W_down: eqx.nn.Linear
    macro_dim: int = eqx.field(static=True)
    micro_dim: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    noise_scale_micro: float = eqx.field(static=True)

    def __init__(self, key: jax.Array, macro_dim: int = 2, micro_dim: int = 10):
        self.macro_dim = macro_dim
        self.micro_dim = micro_dim
        
        self.input_ports = {
            "state": jax.ShapeDtypeStruct((macro_dim + micro_dim,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "precision": jax.ShapeDtypeStruct((), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((macro_dim + micro_dim,), jnp.float32)
        
        # W_down projects from Macro (2D) to Micro (10D)
        self.W_down = eqx.nn.Linear(macro_dim, micro_dim, use_bias=False, key=key)
        
        self.alpha = 0.5
        self.sigma = 0.05
        self.noise_scale_micro = 0.5

    def joint_energy(self, micro: jax.Array, macro: jax.Array, precision: float) -> float:
        # E_local is a multi-well or simple quadratic. We'll use a double-well for rich chaos.
        # E_local = 0.25 * x^4 - 0.5 * x^2
        E_local = jnp.sum(0.25 * micro**4 - 0.5 * micro**2)
        
        # Top-down enslavement prior
        # E_prior = (Precision / 2) * || micro - W_down(macro) ||^2
        prior_target = self.W_down(macro)
        E_prior = 0.5 * precision * jnp.sum((micro - prior_target)**2)
        
        return E_local + E_prior

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        state = inputs["state"]
        dt = inputs["dt"]
        precision = inputs["precision"]
        
        macro = state[:self.macro_dim]
        micro = state[self.macro_dim:]
        
        # Split keys for micro and macro stochastic updates
        key_micro, key_macro = jax.random.split(key)
        
        # 1. Micro Update: Thermodynamic drift via jax.grad on joint Free Energy
        # Note: eqx.filter_value_and_grad can be used, but jax.grad is fine since we just need wrt micro.
        # We must use tree_method or just pass params if we wanted W_down gradients, but we are not training!
        # We are just doing forward physics.
        drift_fn = jax.grad(self.joint_energy, argnums=0)
        drift_micro = -drift_fn(micro, macro, precision)
        
        # Euler-Maruyama step for micro
        diffusion_micro = jnp.sqrt(dt) * self.noise_scale_micro * jax.random.normal(key_micro, micro.shape)
        micro_next = micro + drift_micro * dt + diffusion_micro
        
        # 2. Macro Update: Slow moving average / dampened random walk
        # macro tracks the mean of micro's dimensions (a simple emergent property)
        micro_mean = jnp.mean(micro_next)
        drift_macro = self.alpha * (micro_mean - macro)
        
        diffusion_macro = jnp.sqrt(dt) * self.sigma * jax.random.normal(key_macro, macro.shape)
        macro_next = macro + drift_macro * dt + diffusion_macro
        
        next_state = jnp.concatenate([macro_next, micro_next])
        
        return (next_state, next_state) if return_aux else next_state

    def init_params(self, key):
        return None

def run_simulation(graph, key, precision_val, n_steps, macro_dim, micro_dim):
    # DFG sample with return_aux=True yields the trajectory
    init_macro = jax.random.normal(key, (macro_dim,)) * 0.1
    init_micro = jax.random.normal(key, (micro_dim,)) * 0.1
    initial_state = jnp.concatenate([init_macro, init_micro])
    
    inputs = {
        "initial_state": initial_state,
        "dt": jnp.array(0.05, dtype=jnp.float32),
        "precision": jnp.array(precision_val, dtype=jnp.float32)
    }
    
    _, aux_tup = graph.sample(key, inputs=inputs, params={}, return_aux=True)
    trajectory = aux_tup[0] # Shape: (n_steps, macro_dim + micro_dim)
    
    macro_traj = trajectory[:, :macro_dim]
    micro_traj = trajectory[:, macro_dim:]
    return macro_traj, micro_traj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="jax", help="Backend to use")
    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info(" STABLE BASIN 2.0: TORX HIERARCHICAL ENSLAVEMENT")
    logger.info("="*60)

    device = get_optimal_device(verbose=True, backend=args.backend)
    
    key = jax.random.key(42)
    key, model_key = jax.random.split(key)

    n_steps = 500
    macro_dim = 2
    micro_dim = 10

    # 1. Build the Factor
    enslavement_factor = HierarchicalEnslavementFactor(
        key=model_key, macro_dim=macro_dim, micro_dim=micro_dim
    )
    
    # 2. Wrap in ChainFactor
    chain = ChainFactor(
        base=enslavement_factor,
        n_steps=n_steps,
        feedback_porting_fn="state",
        weight_tied=True
    )
    
    # 3. Build DFG
    graph = DFG(
        sites=(
            Site(
                name="hierarchy",
                factor=chain,
                parents=("initial_state", "dt", "precision"),
                porting_fn=("state", "dt", "precision"),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={
            "initial_state": jax.ShapeDtypeStruct((macro_dim + micro_dim,), jnp.float32),
            "dt": jax.ShapeDtypeStruct((), jnp.float32),
            "precision": jax.ShapeDtypeStruct((), jnp.float32)
        },
        output_name="hierarchy"
    )

    # Compile the graph's sample function for speed
    @jax.jit
    def fast_sample(precision_val):
        # We need a new key for the actual simulation if we wanted independent runs,
        # but using a fixed key for A/B testing gives a clean deterministic comparison.
        return run_simulation(graph, jax.random.key(101), precision_val, n_steps, macro_dim, micro_dim)

    # 4. A/B Testing
    logger.info("[*] Running Regime A: Uncoupled Chaos (Precision = 0.0)")
    macro_A, micro_A = fast_sample(0.0)
    
    logger.info("[*] Running Regime B: Thermodynamic Enslavement (Precision = 10.0)")
    macro_B, micro_B = fast_sample(10.0)
    
    # 5. Variance Analysis
    var_A = jnp.var(micro_A)
    var_B = jnp.var(micro_B)
    reduction = (1.0 - (var_B / var_A)) * 100.0
    
    logger.info(f"\n[+] Variance Reduction Analysis:")
    logger.info(f"    Regime A (Free) Micro Variance:     {var_A:.4f}")
    logger.info(f"    Regime B (Enslaved) Micro Variance: {var_B:.4f}")
    logger.info(f"    -> Thermodynamic Enslavement achieved a {reduction:.2f}% reduction in micro-state entropy.")

    # 6. Dashboard Rendering
    logger.info("\n[*] Rendering Cybernetic Dashboard...")
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    time_axis = jnp.arange(n_steps) * 0.05
    
    # Top Panel
    ax1.plot(time_axis, macro_B[:, 0], 'g-', label="Macro Ch 0", linewidth=2)
    ax1.plot(time_axis, macro_B[:, 1], 'm-', label="Macro Ch 1", linewidth=2)
    ax1.set_title("The Macro Order Parameter (Regime B)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    
    # Middle Panel
    ax2.plot(time_axis, micro_A, alpha=0.5)
    ax2.set_title("Regime A: Uncoupled Chaos (Precision = 0)")
    ax2.set_ylabel("Micro States")
    
    # Bottom Panel
    ax3.plot(time_axis, micro_B, alpha=0.5)
    # Overlay the W_down projection for visual proof
    projected_macro = jax.vmap(enslavement_factor.W_down)(macro_B)
    ax3.plot(time_axis, projected_macro[:, 0], 'w--', linewidth=2, label="Macro Prior Projection (Sample)")
    ax3.set_title(f"Regime B: Thermodynamic Enslavement (Precision = 10) | {reduction:.1f}% Variance Reduction")
    ax3.set_ylabel("Micro States")
    ax3.set_xlabel("Time (s)")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs("output/demo", exist_ok=True)
    plt.savefig("output/demo/torx_hierarchical_enslavement.png", dpi=150)
    logger.info("[SUCCESS] Dashboard rendered and saved to output/demo/torx_hierarchical_enslavement.png")
    logger.info("="*60 + "\n")

if __name__ == "__main__":
    main()
