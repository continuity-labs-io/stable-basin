from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import logging

logger = logging.getLogger(__name__)


class EchoTrainer(eqx.Module):
    """
    Pure functional JAX/Equinox trainer for Backpropagation Through Time (BPTT).
    Aligns a latent Waddington basin to observed biological dynamics via Free Energy minimization.
    """

    optimizer: optax.GradientTransformation
    opt_state: optax.OptState

    def __init__(self, model: eqx.Module, learning_rate: float = 1e-3, max_grad_norm: float = 1.0):
        # Chain gradient clipping and AdamW for gradient stability during BPTT
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adamw(learning_rate)
        )

        # Partition model to initialize the optimizer state only on trainable weights
        trainable, _ = eqx.partition(model, eqx.is_inexact_array)
        self.opt_state = self.optimizer.init(trainable)

        logger.debug("Initialized EchoTrainer with AdamW and global norm clipping.")

    @eqx.filter_jit
    def step(self, model: eqx.Module, batch: dict, key: PRNGKeyArray, dt: float):
        """
        Executes a purely functional JIT-compiled BPTT update step.

        Args:
            model: The Equinox model (e.g., MarkovBlanketObserver).
            batch: Dictionary containing 's_true' (batch, seq_len, d_sensory)
                   and 'x_init' (batch, d_state).
            key: PRNGKeyArray for stochastic unrolling.
            dt: Integration time step.

        Returns:
            new_model, new_trainer, loss_val
        """
        logger.debug("Executing JIT-compiled BPTT update step.")

        # Cleanly separate trainable weights from static topologies
        trainable, static = eqx.partition(model, eqx.is_inexact_array)

        def loss_fn(trainable_model, static_model, s_true_batch, x_init_batch, step_key):
            combined_model = eqx.combine(trainable_model, static_model)

            def single_example_loss(s_true_seq, x_init, ex_key):
                # s_true_seq: (seq_len, d_sensory)
                # We use teacher forcing: inject s_true[0:T-1] to predict x[1:T]
                inputs_seq = s_true_seq[:-1]
                targets_seq = s_true_seq[1:]

                # Execute the physics step via forced unrolling
                predicted_traj = combined_model.forced_unroll(
                    key=ex_key, x_init=x_init, dt=dt, seq=inputs_seq
                )

                # Extract sensory slice of the predicted trajectory
                # Assumes model has a 'hull' defining the Markov Blanket (e.g.
                # MarkovBlanketObserver)
                start_idx = combined_model.hull.d_internal
                end_idx = start_idx + combined_model.hull.d_sensory

                pred_sensory = predicted_traj[:, start_idx:end_idx]

                # Compute MSE strictly between the predicted sensory slice and actual next data
                # frame
                # Do not compute loss on latent internal states
                mse = jnp.mean((pred_sensory - targets_seq) ** 2)
                return mse

            # vmap over batch dimension
            batch_keys = jax.random.split(step_key, s_true_batch.shape[0])
            batch_loss = jax.vmap(single_example_loss)(s_true_batch, x_init_batch, batch_keys)

            return jnp.mean(batch_loss)

        # Compute gradients
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            trainable, static, batch["s_true"], batch["x_init"], key
        )

        # Optimizer update
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, trainable)
        new_trainable = eqx.apply_updates(trainable, updates)

        new_model = eqx.combine(new_trainable, static)

        # Update trainer state immutably
        new_trainer = eqx.tree_at(lambda t: t.opt_state, self, new_opt_state)

        return new_model, new_trainer, loss_val
