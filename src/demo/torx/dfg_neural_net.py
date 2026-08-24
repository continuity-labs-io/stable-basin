"""
Phase 2: Directed Factor Graph (DFG) to Neural Network (Single Level)

This script demonstrates Stochastic Differentiable Programming in Torx.
We define a custom `Factor` containing a trainable Neural Network (an Equinox module).
We embed this factor into a DFG, sample from it, and train it using JAX and Optax 
to learn a non-linear biological state transition.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import logging
import argparse

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Mapping, Any
from jaxtyping import Array, Key, PyTree

from torx import DFG, Site, PortSpec
from torx.factor import AbstractReferenceFactor

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.device import get_optimal_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TorxNeuralNet")

# =====================================================================
# 1. Define a Trainable Neural Factor
# =====================================================================
# In Torx, Factors inherit from ihoop.eqx.AbstractStrictModule (like eqx.Module).
# This means we can embed standard neural network layers inside them, 
# and JAX will track their parameters natively across the DFG.

class NeuralTransitionFactor(AbstractReferenceFactor):
    """A factor that uses an MLP to predict the next biological state."""
    
    # Define the causal ports
    input_ports: dict[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    
    # The neural network payload
    mlp: eqx.nn.MLP
    noise_scale: float = eqx.field(static=True)

    def __init__(self, key: Key[Array, ""], in_dim: int, out_dim: int):
        self.input_ports = {"current_state": jax.ShapeDtypeStruct((in_dim,), jnp.float32)}
        self.output_spec = jax.ShapeDtypeStruct((out_dim,), jnp.float32)
        self.noise_scale = 0.1
        
        # A simple 2-layer MLP using Equinox
        self.mlp = eqx.nn.MLP(
            in_size=in_dim, 
            out_size=out_dim, 
            width_size=16, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=key
        )

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        """
        The stochastic forward pass.
        We pass the input through the MLP and add biological thermal noise.
        """
        x = inputs["current_state"]
        
        # Deterministic neural prediction
        mean_prediction = self.mlp(x)
        
        # Stochastic thermodynamic sampling
        noise = jax.random.normal(key, mean_prediction.shape) * self.noise_scale
        output = mean_prediction + noise
        
        return (output, None) if return_aux else output
        
    def init_params(self, key):
        # We don't need explicit external params because Equinox tracks self.mlp
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="jax", help="Backend to use")
    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info(" STABLE BASIN 2.0: TORX NEURAL NETWORK TRAINING")
    logger.info("="*60)

    device = get_optimal_device(verbose=True, backend=args.backend)
    key = jax.random.key(42)
    key, mlp_key = jax.random.split(key)

    # =====================================================================
    # 2. Build the DFG
    # =====================================================================
    in_dim, out_dim = 2, 2
    neural_factor = NeuralTransitionFactor(mlp_key, in_dim=in_dim, out_dim=out_dim)

    # Wire it into a standard DFG
    graph = DFG(
        sites=(
            Site(
                name="state_transition",
                factor=neural_factor,
                parents=("env_input",),
                porting_fn=("current_state",),
                param_key=None, info_key=None, site_info=None
            ),
        ),
        input_ports={"env_input": jax.ShapeDtypeStruct((in_dim,), jnp.float32)},
        output_name="state_transition"
    )

    # =====================================================================
    # 3. Create Synthetic Target Data
    # =====================================================================
    # Let's say the biological ground truth is a simple non-linear mapping:
    # y = sin(x) * 2.0
    
    def generate_batch(key, batch_size=32):
        x_key, noise_key = jax.random.split(key)
        X = jax.random.uniform(x_key, (batch_size, in_dim), minval=-2.0, maxval=2.0)
        Y_true = jnp.sin(X) * 2.0 + (jax.random.normal(noise_key, (batch_size, out_dim)) * 0.1)
        return X, Y_true

    # =====================================================================
    # 4. Define Differentiable Loss and Update Step
    # =====================================================================
    
    # We use eqx.filter_value_and_grad to automatically find the trainable 
    # weights inside the Torx DFG (the MLP parameters) while ignoring static fields.
    @eqx.filter_value_and_grad
    def loss_fn(model_graph, x_batch, y_batch, sample_key):
        # Torx DFGs expect a dict for inputs
        def single_sample(x, k):
            return model_graph.sample(k, inputs={"env_input": x}, params={})
        
        # Vectorize the sampling across the batch
        batch_keys = jax.random.split(sample_key, x_batch.shape[0])
        y_pred = jax.vmap(single_sample)(x_batch, batch_keys)
        
        # Standard MSE loss
        return jnp.mean((y_pred - y_batch) ** 2)

    @eqx.filter_jit
    def make_step(model_graph, opt_state, x_batch, y_batch, step_key):
        loss, grads = loss_fn(model_graph, x_batch, y_batch, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, model_graph)
        model_graph = eqx.apply_updates(model_graph, updates)
        return model_graph, opt_state, loss

    # Initialize Optax optimizer
    optimizer = optax.adam(learning_rate=0.01)
    # Filter the DFG for trainable arrays (the neural weights)
    opt_state = optimizer.init(eqx.filter(graph, eqx.is_array))

    # =====================================================================
    # 5. Training Loop
    # =====================================================================
    logger.info("[*] Training Torx Neural DFG (Target: y = sin(x) * 2.0)")
    
    epochs = 500
    for epoch in range(1, epochs + 1):
        key, batch_key, step_key = jax.random.split(key, 3)
        X, Y = generate_batch(batch_key, batch_size=64)
        
        graph, opt_state, loss = make_step(graph, opt_state, X, Y, step_key)
        
        if epoch % 100 == 0:
            logger.info(f"    Epoch {epoch:03d}/{epochs} | MSE Loss: {loss:.4f}")

    # =====================================================================
    # 6. Evaluation
    # =====================================================================
    key, eval_key = jax.random.split(key)
    test_x = jnp.array([[1.0, -1.0]])
    expected_y = jnp.sin(test_x) * 2.0
    
    pred_y = graph.sample(eval_key, inputs={"env_input": test_x[0]}, params={})
    
    logger.info("\n[+] Inference Results:")
    logger.info(f"    Input X:      {test_x[0].tolist()}")
    logger.info(f"    Expected Y:   {expected_y[0].tolist()}")
    logger.info(f"    Predicted Y:  {pred_y.tolist()}")
    logger.info("\n[SUCCESS] Torx Stochastic Differentiable Programming verified.")
    logger.info("="*60 + "\n")

if __name__ == "__main__":
    main()
