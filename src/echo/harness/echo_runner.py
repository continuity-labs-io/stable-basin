from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import wandb
import yaml
from typing import Optional

try:
    import ray.train

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from src.echo.harness.echo_trainer import EchoTrainer
from src.echo.metrics.energy_landscape import batch_hessian_trace, ScalarEnergy
from src.harness.pytorch_jax_bridge import torch_to_jax

logger = logging.getLogger(__name__)


@eqx.filter_jit
def compute_validation_loss(
    model: eqx.Module,
    s_true_batch: jax.Array,
    x_init_batch: jax.Array,
    key: PRNGKeyArray,
    dt: float,
) -> jax.Array:
    """
    Computes teacher-forced validation MSE loss.
    """

    def single_example_loss(s_true_seq, x_init, ex_key):
        inputs_seq = s_true_seq[:-1]
        targets_seq = s_true_seq[1:]

        predicted_traj = model.forced_unroll(key=ex_key, x_init=x_init, dt=dt, seq=inputs_seq)

        start_idx = model.hull.d_internal
        end_idx = start_idx + model.hull.d_sensory

        pred_sensory = predicted_traj[:, start_idx:end_idx]
        return jnp.mean((pred_sensory - targets_seq) ** 2)

    batch_keys = jax.random.split(key, s_true_batch.shape[0])
    batch_loss = jax.vmap(single_example_loss)(s_true_batch, x_init_batch, batch_keys)
    return jnp.mean(batch_loss)


class EchoRunner:
    """
    Orchestrates the training lifecycle, managing PyTorch DataLoaders and the pure Equinox/Optax
        training loop.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.trainer: Optional[EchoTrainer] = None
        self.wandb_run = None

    def setup(self, trainer: EchoTrainer):
        """
        Initializes W&B and stores the trainer.
        """
        self.trainer = trainer

        wandb_project = self.config.get("logging", {}).get("wandb_project")
        if wandb_project and wandb.run is None:
            self.wandb_run = wandb.init(project=wandb_project, config=self.config)
            logger.info(f"Initialized W&B project: {wandb_project}")

    def train_epoch(self, model: eqx.Module, train_loader, key: PRNGKeyArray, dt: float):
        """
        Executes a single training epoch.
        """
        if self.trainer is None:
            raise RuntimeError("EchoRunner.setup(trainer) must be called before training.")

        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            jax_batch = {
                "s_true": torch_to_jax(batch["s_true"]),
                "x_init": torch_to_jax(batch["x_init"]),
            }

            key, step_key = jax.random.split(key)
            model, self.trainer, loss_val = self.trainer.step(model, jax_batch, step_key, dt)

            loss_scalar = loss_val.item()
            epoch_losses.append(loss_scalar)

            if batch_idx % 10 == 0:
                logger.info(f"Train Batch {batch_idx} Loss: {loss_scalar:.4f}")

        return model, sum(epoch_losses) / len(epoch_losses)

    def validate(self, model: eqx.Module, val_loader, key: PRNGKeyArray, dt: float):
        """
        Executes a validation pass, calculating standard MSE loss and Hessian trace curvature.
        """
        val_losses = []
        hessian_traces = []
        max_hessian_samples = 32
        hessian_computed = 0
        energy_fn = ScalarEnergy(model.ebm, model.hull)

        for batch_idx, batch in enumerate(val_loader):
            s_true = torch_to_jax(batch["s_true"])
            x_init = torch_to_jax(batch["x_init"])

            key, val_key = jax.random.split(key)
            loss_val = compute_validation_loss(model, s_true, x_init, val_key, dt)
            val_losses.append(loss_val.item())

            # Compute Hessian trace on a small fixed-size subset for memory safety
            if hessian_computed < max_hessian_samples:
                samples_to_take = min(s_true.shape[0], max_hessian_samples - hessian_computed)
                x_subset = x_init[:samples_to_take]

                trace_batch = batch_hessian_trace(energy_fn, x_subset)
                trace_val = jnp.mean(trace_batch)
                hessian_traces.append(trace_val.item())
                hessian_computed += samples_to_take

        val_loss_scalar = sum(val_losses) / len(val_losses) if val_losses else 0.0
        trace_scalar = sum(hessian_traces) / len(hessian_traces) if hessian_traces else 0.0

        return val_loss_scalar, trace_scalar

    def run(self, model: eqx.Module, train_loader, val_loader, key: PRNGKeyArray, dt: float):
        """
        Executes the full training loop over multiple epochs based on configuration.
        """
        max_epochs = self.config.get("optimization", {}).get("max_epochs", 1)
        learning_rate = self.config.get("optimization", {}).get("learning_rate", 1e-3)
        patience = self.config.get("optimization", {}).get("early_stopping_patience", None)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            key, train_key, val_key = jax.random.split(key, 3)

            model, train_loss = self.train_epoch(model, train_loader, train_key, dt)
            val_loss, hessian_trace = self.validate(model, val_loader, val_key, dt)

            logger.info(
                f"Epoch {epoch + 1} complete. Validation loss: {val_loss:.4f}. Hessian "
                f"trace: {hessian_trace:.4f}."
            )

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "hessian_trace": hessian_trace,
                "learning_rate": learning_rate,
                "epoch": epoch + 1,
            }

            if wandb.run is not None:
                wandb.log(metrics)

            if RAY_AVAILABLE:
                try:
                    if ray.train.get_context().get_trial_name():
                        ray.train.report(metrics)
                except RuntimeError:
                    pass
                except Exception as e:
                    logger.debug(f"Ray train report failed: {e}")

            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(
                            f"Early stopping at epoch {epoch + 1}, patience={patience}."
                        )
                        break

        return model
