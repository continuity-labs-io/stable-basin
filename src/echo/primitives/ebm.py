import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Tuple

class PrecisionWeightedEBM(eqx.Module):
    """
    Core Energy-Based Model for the biological observer.
    Maps a biological state vector `x` to two outputs:
    1. A scalar Energy E_θ(x) representing the thermodynamic potential.
    2. A Precision matrix Π_θ(x) (SPD) representing the certainty/steepness
       of the local energy landscape.
    """
    mlp: eqx.nn.MLP
    energy_head: eqx.nn.Linear
    precision_head: eqx.nn.Linear
    d_state: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        d_state: int,
        hidden_size: int,
        depth: int,
        key: PRNGKeyArray,
        epsilon: float = 1e-4
    ):
        """
        Initializes the PrecisionWeightedEBM.
        
        Args:
            d_state: Dimensionality of the input state.
            hidden_size: Number of hidden units in the MLP layers.
            depth: Number of hidden layers in the MLP backbone.
            key: PRNGKey for initialization.
            epsilon: Diagonal jitter for ensuring the precision matrix is strictly SPD.
        """
        self.d_state = d_state
        self.epsilon = epsilon
        
        key_mlp, key_energy, key_precision = jax.random.split(key, 3)
        
        # Backbone MLP. Must use a smooth activation function (e.g., GELU)
        # to ensure the network is twice-differentiable everywhere.
        self.mlp = eqx.nn.MLP(
            in_size=d_state,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=key_mlp
        )
        
        # Energy head: maps from hidden state to 1 scalar feature
        self.energy_head = eqx.nn.Linear(hidden_size, 1, key=key_energy)
        
        # Precision head: maps from hidden state to d_state * d_state features
        self.precision_head = eqx.nn.Linear(
            hidden_size,
            d_state * d_state,
            key=key_precision
        )

    def __call__(self, x: Float[Array, "d_state"]) -> Tuple[Float[Array, ""], Float[Array, "d_state d_state"]]:
        """
        Forward pass mapping state vector `x` to (energy, precision).
        
        Args:
            x: 1D state vector of shape (d_state,).
            
        Returns:
            A tuple of (energy, precision) where:
            - energy is a scalar JAX array of shape ().
            - precision is an SPD matrix of shape (d_state, d_state).
        """
        # 1. Process through the backbone MLP
        h = self.mlp(x)
        
        # 2. Compute scalar energy
        energy_raw = self.energy_head(h)
        # Bound energy from below to ensure a thermodynamic floor (prevents infinite sinkholes)
        e_mlp = jnp.squeeze(jax.nn.softplus(energy_raw))  # Shape: ()
        
        # Add a global structural prior to guarantee the landscape is a positive-definite basin
        e_prior = 0.5 * 0.1 * jnp.sum(x ** 2)
        energy = e_prior + e_mlp
        
        # 3. Compute precision matrix
        precision_flat = self.precision_head(h)
        # Smoothly bound the raw precision to prevent exponential blowup during long BPTT unrolls
        precision_flat = jnp.tanh(precision_flat) * 5.0
        # Reshape to (d_state, d_state)
        W_raw = precision_flat.reshape((self.d_state, self.d_state))
        
        # Enforce strictly symmetric positive-definite via Cholesky decomposition
        L = jnp.tril(W_raw)
        precision = (L @ L.T) + (self.epsilon * jnp.eye(self.d_state, dtype=jnp.float32))
        
        return energy, precision

class GaussianEBM(eqx.Module):
    """
    Implements a rigid, single-basin parabolic landscape: E(x) = 1/2 * (x - mu)^T Pi (x - mu).
    Pi is parameterized via Cholesky decomposition (L L^T) and is constant everywhere.
    """
    mu: Float[Array, "d_state"]
    L: Float[Array, "d_state d_state"]
    d_state: int = eqx.field(static=True)

    def __init__(self, d_state: int, hidden_size: int, depth: int, key: PRNGKeyArray, epsilon: float = 1e-4):
        """
        Initializes the GaussianEBM.
        (hidden_size and depth are ignored, kept for API compatibility with PrecisionWeightedEBM)
        """
        self.d_state = d_state
        k1, k2 = jax.random.split(key)
        self.mu = jax.random.normal(k1, (d_state,))
        # Initialize L to be roughly identity so Pi is roughly identity
        self.L = jax.random.normal(k2, (d_state, d_state)) * 0.1 + jnp.eye(d_state)

    def __call__(self, x: Float[Array, "d_state"]) -> Tuple[Float[Array, ""], Float[Array, "d_state d_state"]]:
        L_tril = jnp.tril(self.L)
        Pi = L_tril @ L_tril.T
        diff = x - self.mu
        energy = 0.5 * jnp.dot(diff, jnp.dot(Pi, diff))
        return energy, Pi

