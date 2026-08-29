import jax
import jax.numpy as jnp
from functools import partial
import torch
from torch.utils.data import Dataset
import numpy as np

# Standard coefficients for Muller-Brown potential
A = jnp.array([-200.0, -100.0, -170.0, 15.0])
a = jnp.array([-1.0, -1.0, -6.5, 0.7])
b = jnp.array([0.0, 0.0, 11.0, 0.6])
c = jnp.array([-10.0, -10.0, -6.5, 0.7])
x0 = jnp.array([1.0, 0.0, -0.5, -1.0])
y0 = jnp.array([0.0, 0.5, 1.5, 1.0])

def muller_brown_potential(state: jax.Array) -> jax.Array:
    """
    Computes the Müller-Brown potential at a given 2D state [x, y].
    """
    x, y = state[0], state[1]
    
    term1 = a * (x - x0)**2
    term2 = b * (x - x0) * (y - y0)
    term3 = c * (y - y0)**2
    
    V = jnp.sum(A * jnp.exp(term1 + term2 + term3))
    return V

@partial(jax.jit, static_argnums=(2,))
def generate_trajectory(key: jax.random.PRNGKey, state_init: jax.Array, n_steps: int, dt: float, kT: float) -> jax.Array:
    """
    Unrolls overdamped Langevin dynamics on the Müller-Brown potential using Euler-Maruyama.
    """
    def step_fn(state, key_step):
        # Compute analytical gradient
        grad = jax.grad(muller_brown_potential)(state)
        
        # Apply gradient clipping to prevent explosion in steep regions
        grad = jnp.clip(grad, -100.0, 100.0)
        
        # Brownian noise
        dW = jax.random.normal(key_step, shape=state.shape, dtype=jnp.float32)
        
        # Euler-Maruyama update
        next_state = state - grad * dt + jnp.sqrt(2.0 * kT * dt) * dW
        
        return next_state, next_state

    keys = jax.random.split(key, n_steps)
    _, trajectory = jax.lax.scan(step_fn, state_init, keys)
    return trajectory


class MullerBrownDataset(Dataset):
    """
    A PyTorch-compatible dataset generating continuous-time Langevin trajectories
    on the Müller-Brown potential using fast JAX simulation.
    """
    def __init__(self, size: int, seq_len: int, dt: float = 0.001, kT: float = 15.0, seed: int = 42):
        self.size = size
        self.seq_len = seq_len
        self.dt = dt
        self.kT = kT
        
        key = jax.random.PRNGKey(seed)
        key, init_key, sim_key = jax.random.split(key, 3)
        
        # Spawn particles near the three known minima
        centers = jnp.array([
            [-0.5, 1.5],
            [0.0, 0.5],
            [0.5, 0.0]
        ])
        
        # Randomly assign a center to each sequence
        center_idx_key, noise_key = jax.random.split(init_key, 2)
        center_idxs = jax.random.choice(center_idx_key, 3, shape=(size,))
        chosen_centers = centers[center_idxs]
        
        # Add small Gaussian noise to initial positions
        initial_states = chosen_centers + 0.1 * jax.random.normal(noise_key, shape=(size, 2))
        
        # VMAP the trajectory generation
        sim_keys = jax.random.split(sim_key, size)
        
        # jax.vmap signature: in_axes=(0, 0, None, None, None)
        batched_generate = jax.vmap(generate_trajectory, in_axes=(0, 0, None, None, None))
        
        # Generate all trajectories at once
        trajectories = batched_generate(sim_keys, initial_states, seq_len, dt, kT)
        
        # Convert to PyTorch tensors
        self.data = torch.from_numpy(np.array(trajectories)).float()
        self.mask = torch.ones_like(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        mask = self.mask[idx]
        return {
            "x_raw": x,
            "mask": mask,
            "y_true": x.clone()
        }
