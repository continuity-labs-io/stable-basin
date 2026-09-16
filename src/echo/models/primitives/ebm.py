import logging
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

logger = logging.getLogger(__name__)


class GaussianEBM(eqx.Module):
    """
    Implements a rigid, single-basin parabolic landscape: E(x) = 1/2 * (x - mu)^T Pi (x - mu).
    Pi is parameterized via Cholesky decomposition (L L^T).
    """
    mu: Float[Array, "dim"]
    L: Float[Array, "dim dim"]

    def __init__(self, dim: int, key: jax.Array):
        """
        Initializes the GaussianEBM.

        Args:
            dim: The dimensionality of the state vector.
            key: JAX PRNG key for initialization.
        """
        k1, k2 = jax.random.split(key)
        self.mu = jax.random.normal(k1, (dim,))
        self.L = jax.random.normal(k2, (dim, dim)) * 0.1 + jnp.eye(dim)

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        """
        Computes the energy of the given state vector.

        Args:
            x: The input state vector.

        Returns:
            The scalar energy value.
        """
        L_tril = jnp.tril(self.L)
        Pi = L_tril @ L_tril.T
        diff = x - self.mu
        energy = 0.5 * jnp.dot(diff, jnp.dot(Pi, diff))
        return energy


class PrecisionWeightedEBM(eqx.Module):
    """
    Implements a learned, free-form energy landscape capable of asymmetric,
    multimodal Waddington wells.
    """
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, dim: int, hidden_dim: int, key: jax.Array):
        """
        Initializes the PrecisionWeightedEBM.

        Args:
            dim: The dimensionality of the state vector.
            hidden_dim: The hidden dimension of the MLP.
            key: JAX PRNG key for initialization.
        """
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(dim, hidden_dim, key=k1)
        self.layer2 = eqx.nn.Linear(hidden_dim, 1, key=k2)

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        """
        Computes the energy of the given state vector.

        Args:
            x: The input state vector.

        Returns:
            The scalar energy value.
        """
        h = jax.nn.silu(self.layer1(x))
        energy = jax.nn.silu(self.layer2(h))
        return jnp.squeeze(energy)
