import os
import wandb
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import yaml
import json
import logging
from torch.utils.data import DataLoader
import torch

from src.data.behavior.celegans_gait_dataset import RealEigenwormDataset
from src.data.datasets import JAXDictDataset
from src.benchmarks.worm_gait.core import build_graph
from src.echo.primitives.ebm import PrecisionWeightedEBM
from src.echo.architecture.hierarchical_factor import HierarchicalThermoFlowFactor
from src.echo.architecture.predictive_coding_graph import PredictiveCodingGraph
from src.echo.primitives.thermalizer import ForcedTorxThermalizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_next(graph: PredictiveCodingGraph, log_lambda: jax.Array, x: jax.Array, seq_frame: jax.Array, dt: float, key: jax.Array):
    factor = graph.flow_factor
    
    d_internal = graph.hull.d_internal
    x_injected = jax.lax.dynamic_update_slice(x, seq_frame, (d_internal,))
    
    x_micro = x_injected[: factor.d_micro]
    x_macro = x_injected[factor.d_micro :]
    
    grad_micro, grad_macro = jax.grad(factor.joint_energy_fn, argnums=(0, 1))(x_micro, x_macro)
    
    lambda_val = jnp.exp(log_lambda)
    grad_micro = grad_micro * lambda_val
    grad_macro = grad_macro * lambda_val
    
    params = factor.precompute()
    Q_micro_masked = params.get("Q_micro_masked", factor.micro_solenoidal.Q)
    Gamma_micro_masked = params.get("Gamma_micro_masked", factor.micro_dissipative.Gamma)
    Q_macro_masked = params.get("Q_macro_masked", factor.macro_solenoidal.Q)
    Gamma_macro_masked = params.get("Gamma_macro_masked", factor.macro_dissipative.Gamma)
    S_micro = params["S_micro"]
    S_macro = params["S_macro"]
    
    k_micro, k_macro = jax.random.split(key, 2)
    
    q_micro = jnp.zeros(factor.d_micro, dtype=jnp.float32)
    omega_micro = jnp.zeros(factor.d_micro, dtype=jnp.float32)
    q_macro = jnp.zeros(factor.d_macro, dtype=jnp.float32)
    omega_macro = jnp.zeros(factor.d_macro, dtype=jnp.float32)
    
    x_micro_next = factor.micro_thermostat(
        x=x_micro,
        grad_E=grad_micro,
        Q=Q_micro_masked,
        L=jax.lax.stop_gradient(S_micro),
        dt=dt,
        key=k_micro,
        omega_ext=omega_micro,
        q_ext=q_micro,
        Gamma=Gamma_micro_masked,
    )

    x_macro_next = factor.macro_thermostat(
        x=x_macro,
        grad_E=grad_macro,
        Q=Q_macro_masked,
        L=jax.lax.stop_gradient(S_macro),
        dt=dt,
        key=k_macro,
        omega_ext=omega_macro,
        q_ext=q_macro,
        Gamma=Gamma_macro_masked,
    )
    
    x_next = jnp.concatenate([x_micro_next, x_macro_next])
    return x_next


def forced_unroll_1step(graph: PredictiveCodingGraph, log_lambda: jax.Array, x_init: jax.Array, seq: jax.Array, dt: float, key: jax.Array):
    seq_len = seq.shape[0]
    keys = jax.random.split(key, seq_len)
    
    def step_fn(state, carry):
        seq_frame, step_key = carry
        next_state = predict_next(graph, log_lambda, state, seq_frame, dt, step_key)
        return next_state, next_state
        
    _, traj = jax.lax.scan(step_fn, x_init, (seq, keys))
    return traj


class LambdaModel(eqx.Module):
    log_lambda: jax.Array
    
    def __init__(self, log_lambda_init: float):
        self.log_lambda = jnp.array(log_lambda_init)

    def __call__(self, graph: PredictiveCodingGraph, x_init: jax.Array, seq: jax.Array, dt: float, key: jax.Array):
        return forced_unroll_1step(graph, self.log_lambda, x_init, seq, dt, key)


@eqx.filter_value_and_grad
def loss_fn(lambda_model: LambdaModel, graph: PredictiveCodingGraph, x_init: jax.Array, s_true: jax.Array, dt: float, key: jax.Array):
    seq_in = s_true[:-1]
    s_target = s_true[1:]
    
    traj = lambda_model(graph, x_init, seq_in, dt, key)
    
    d_internal = graph.hull.d_internal
    d_sensory = graph.hull.d_sensory
    pred_sensory = traj[:, d_internal : d_internal + d_sensory]
    
    mse = jnp.mean((pred_sensory - s_target) ** 2)
    return mse


@eqx.filter_jit
def train_step(lambda_model: LambdaModel, graph: PredictiveCodingGraph, x_init_batch: jax.Array, s_true_batch: jax.Array, dt: float, key: jax.Array, opt_state, optim):
    batch_size = x_init_batch.shape[0]
    keys = jax.random.split(key, batch_size)
    
    def batch_loss(model):
        losses = jax.vmap(lambda x, s, k: loss_fn(model, graph, x, s, dt, k)[0])(x_init_batch, s_true_batch, keys)
        return jnp.mean(losses)
    
    loss, grads = eqx.filter_value_and_grad(batch_loss)(lambda_model)
    
    updates, opt_state = optim.update(grads, opt_state, lambda_model)
    lambda_model = eqx.apply_updates(lambda_model, updates)
    
    return lambda_model, opt_state, loss


def main():
    config_path = "configs/worm_gait_ebm.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    dt = config.get("experiment", {}).get("dt", 0.01)

    logger.info("Loading Old Worm dataset...")
    seq_len = 100
    try:
        eval_old_dataset_raw = RealEigenwormDataset(
            data_path="data/worm/EigenWorms_TEST.ts", seq_len=seq_len, is_aged=True
        )
    except FileNotFoundError:
        logger.error("Biological data not found. Ensure EigenWorms_TEST.ts exists.")
        return

    key, subkey = jax.random.split(key)
    dummy_graph, d_state = build_graph(PrecisionWeightedEBM, subkey, config)

    dataset = JAXDictDataset(eval_old_dataset_raw, d_state)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model_path = "output/benchmarks/worm_gait/06_worm_gait_decline_trained_engine.eqx"
    if not os.path.exists(model_path):
        logger.error(f"Pre-trained model not found at {model_path}. Please run benchmark 06 first.")
        return

    graph = eqx.tree_deserialise_leaves(model_path, dummy_graph)
    
    lambda_model = LambdaModel(log_lambda_init=0.0)
    
    optim = optax.adam(learning_rate=0.01)
    opt_state = optim.init(eqx.filter(lambda_model, eqx.is_inexact_array))

    logger.info("Starting inference optimization for log_lambda...")
    
    epochs = 30
    wandb.init(project="worm_gait", name="02_infer_biological_lambda")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            x_init = batch["x_init"].numpy()
            s_true = batch["s_true"].numpy()
            
            key, subkey = jax.random.split(key)
            lambda_model, opt_state, loss = train_step(
                lambda_model, graph, jnp.array(x_init), jnp.array(s_true), dt, subkey, opt_state, optim
            )
            epoch_loss += loss.item()
            batches += 1
            
        current_lambda = jnp.exp(lambda_model.log_lambda)
        wandb.log({"train_loss": epoch_loss/batches, "inferred_lambda": float(current_lambda), "epoch": epoch})
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/batches:.4f}, Inferred lambda: {current_lambda:.4f}")

    final_lambda = float(jnp.exp(lambda_model.log_lambda))
    logger.info(f"Optimization complete. Final inferred biological lambda: {final_lambda:.4f}")

    out_path = "output/benchmarks/worm_gait/02_inferred_biological_lambda.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"biological_lambda": final_lambda}, f, indent=4)
        
    wandb.log({"final_biological_lambda": final_lambda})
    
    artifact = wandb.Artifact("02_biological_lambda_json", type="metrics")
    artifact.add_file(out_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    logger.info(f"Saved result to {out_path} and logged to wandb")


if __name__ == "__main__":
    main()
