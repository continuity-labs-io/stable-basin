"""
Phase 3: Torx Observer Zero Port (Macro-State Extraction)

Demonstrates extracting a slow, stable macroscopic order parameter from chaotic 
micro-states using a Torx probabilistic Directed Factor Graph (DFG).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import logging
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torx import DFG, Site, ChainFactor
from torx.factor import AbstractReferenceFactor

from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TorxObserverZero")

class MacroObserverFactor(AbstractReferenceFactor):
    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)
    
    mlp: eqx.nn.MLP
    noise_scale: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, in_dim: int = 16, out_dim: int = 2, seq_len: int = 200):
        self.seq_len = seq_len
        self.input_ports = {
            "micro_state": jax.ShapeDtypeStruct((seq_len, in_dim), jnp.float32),
            "prev_macro_state": jax.ShapeDtypeStruct((out_dim + 1,), jnp.float32)
        }
        self.output_spec = jax.ShapeDtypeStruct((out_dim + 1,), jnp.float32)
        
        self.mlp = eqx.nn.MLP(in_size=in_dim, out_size=out_dim, width_size=32, depth=1, activation=jax.nn.gelu, key=key)
        self.noise_scale = 0.01
        self.alpha = 0.05

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        micro_seq = inputs["micro_state"]
        prev_macro_full = inputs["prev_macro_state"]
        
        prev_macro = prev_macro_full[:2]
        step = prev_macro_full[2].astype(jnp.int32)
        
        # Clamp to prevent out-of-bounds during JIT/unrolling
        step = jnp.clip(step, 0, self.seq_len - 1)
        
        current_micro = micro_seq[step]
        
        # Deterministic projection
        mean_prediction = self.mlp(current_micro)
        
        # Stochastic thermodynamic sampling
        noise = jax.random.normal(key, prev_macro.shape) * self.noise_scale
        
        # Leaky integrator (thermodynamic low-pass filter)
        next_macro = (1 - self.alpha) * prev_macro + self.alpha * mean_prediction + noise
        
        # Increment step counter and concatenate
        next_macro_full = jnp.concatenate([next_macro, jnp.array([step + 1], dtype=jnp.float32)])
        
        return (next_macro_full, next_macro) if return_aux else next_macro_full

    def init_params(self, key):
        return None

def generate_data(key, batch_size=32, seq_len=200, in_dim=16, out_dim=2):
    t = jnp.linspace(0, 4 * jnp.pi, seq_len)
    
    # Ground Truth Macro: slow, clean, low-frequency sine wave (2D for 2 channels)
    macro_true_1 = jnp.sin(t)
    macro_true_2 = jnp.cos(t)
    Y_true = jnp.stack([macro_true_1, macro_true_2], axis=-1)  # (seq_len, 2)
    Y_true_batch = jnp.tile(Y_true[None, ...], (batch_size, 1, 1))  # (batch_size, seq_len, 2)
    
    # Observed Micro: 16-D projection + high frequency noise
    proj_key, noise_key = jax.random.split(key)
    # Random projection matrix
    W = jax.random.normal(proj_key, (out_dim, in_dim))
    
    # Project macro to micro
    X_clean = Y_true_batch @ W
    
    # Add large-variance Gaussian noise
    noise = jax.random.normal(noise_key, X_clean.shape) * 2.0
    X_noisy = X_clean + noise  # (batch_size, seq_len, 16)
    
    return X_noisy, Y_true_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="jax", help="Backend to use")
    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info(" STABLE BASIN 2.0: TORX OBSERVER ZERO PORT")
    logger.info("="*60)

    device = get_optimal_device(verbose=True, backend=args.backend)
    
    key = jax.random.key(42)
    key, mlp_key = jax.random.split(key)

    seq_len = 200
    in_dim = 16
    out_dim = 2
    batch_size = 32

    # 1. Build the MacroObserverFactor
    observer_factor = MacroObserverFactor(key=mlp_key, in_dim=in_dim, out_dim=out_dim, seq_len=seq_len)
    
    # 2. Wrap in ChainFactor
    chain = ChainFactor(
        base=observer_factor,
        n_steps=seq_len,
        feedback_porting_fn="prev_macro_state",
        weight_tied=True
    )
    
    # 3. Build DFG
    graph = DFG(
        sites=(
            Site(
                name="observer",
                factor=chain,
                parents=("env_micro", "init_macro"),
                porting_fn=("micro_state", "prev_macro_state"),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={
            "env_micro": jax.ShapeDtypeStruct((seq_len, in_dim), jnp.float32),
            "init_macro": jax.ShapeDtypeStruct((out_dim + 1,), jnp.float32)
        },
        output_name="observer"
    )

    # 4. Define Loss Function
    @eqx.filter_value_and_grad
    def loss_fn(model_graph, x_batch, y_batch, sample_key):
        def single_sample(x, k):
            init_macro = jnp.zeros(out_dim + 1, dtype=jnp.float32)
            _, aux_tup = model_graph.sample(k, inputs={"env_micro": x, "init_macro": init_macro}, params={}, return_aux=True)
            return aux_tup[0]
            
        batch_keys = jax.random.split(sample_key, x_batch.shape[0])
        y_pred = jax.vmap(single_sample)(x_batch, batch_keys)
        
        mse_loss = jnp.mean((y_pred - y_batch) ** 2)
        slowness_penalty = jnp.mean((y_pred[:, 1:, :] - y_pred[:, :-1, :]) ** 2)
        
        return mse_loss + 0.1 * slowness_penalty

    @eqx.filter_jit
    def make_step(model_graph, opt_state, x_batch, y_batch, step_key):
        loss, grads = loss_fn(model_graph, x_batch, y_batch, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, model_graph)
        model_graph = eqx.apply_updates(model_graph, updates)
        return model_graph, opt_state, loss

    optimizer = optax.adamw(learning_rate=0.01)
    opt_state = optimizer.init(eqx.filter(graph, eqx.is_array))

    logger.info("[*] Generating Synthetic Data...")
    key, data_key = jax.random.split(key)
    X, Y = generate_data(data_key, batch_size=batch_size, seq_len=seq_len, in_dim=in_dim, out_dim=out_dim)

    logger.info("[*] Training Torx Observer Zero...")
    epochs = 300
    for epoch in range(1, epochs + 1):
        key, step_key = jax.random.split(key)
        graph, opt_state, loss = make_step(graph, opt_state, X, Y, step_key)
        
        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"    Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f}")

    logger.info("\n[*] Rendering Dashboard...")
    key, eval_key = jax.random.split(key)
    
    test_x = X[0]
    test_y = Y[0]
    
    init_macro = jnp.zeros(out_dim + 1, dtype=jnp.float32)
    _, aux_tup = graph.sample(eval_key, inputs={"env_micro": test_x, "init_macro": init_macro}, params={}, return_aux=True)
    pred_y = aux_tup[0]
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(test_x, alpha=0.3)
    ax1.set_title("The Micro-State Chaos (16-D Noisy Inputs)")
    ax1.set_ylabel("Amplitude")
    
    ax2.plot(test_y[:, 0], 'g--', label="Ground Truth (Ch 0)")
    ax2.plot(test_y[:, 1], 'm--', label="Ground Truth (Ch 1)")
    ax2.plot(pred_y[:, 0], 'g-', label="Extracted Macro (Ch 0)", linewidth=2)
    ax2.plot(pred_y[:, 1], 'm-', label="Extracted Macro (Ch 1)", linewidth=2)
    ax2.set_title("The Macro-State Enslavement (Extracted 2-D Order Parameter)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs("output/demo", exist_ok=True)
    plt.savefig("output/demo/torx_observer_zero.png", dpi=150)
    logger.info("[SUCCESS] Dashboard rendered and saved to output/demo/torx_observer_zero.png")
    logger.info("="*60 + "\n")

if __name__ == "__main__":
    main()
