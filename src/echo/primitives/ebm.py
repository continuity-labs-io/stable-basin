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
        energy = jnp.squeeze(energy_raw)  # Shape: ()
        
        # 3. Compute precision matrix
        precision_flat = self.precision_head(h)
        # Reshape to (d_state, d_state)
        W_raw = precision_flat.reshape((self.d_state, self.d_state))
        
        # Enforce strictly symmetric positive-definite via Cholesky decomposition
        L = jnp.tril(W_raw)
        precision = (L @ L.T) + (self.epsilon * jnp.eye(self.d_state, dtype=jnp.float32))
        
        return energy, precision
