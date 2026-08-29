import math
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray

class DissipativeFriction(eqx.Module):
    """
    Computes a strictly symmetric positive-definite friction matrix Γ,
    parameterized via a Cholesky-style lower-triangular factorization:
    Γ = (L @ L^T) + (epsilon * I).
    
    This represents the energy-consuming homeostatic correction (the "brakes"
    that pull the system down the gradient).
    """
    W: Float[Array, "d_state d_state"]
    epsilon: float

    def __init__(self, d_state: int, key: PRNGKeyArray, epsilon: float = 1e-4):
        """
        Initializes the DissipativeFriction module.
        
        Args:
            d_state: The dimensionality of the state vector.
            key: PRNG key for initialization.
            epsilon: Jitter added to the diagonal to ensure strict positive-definiteness.
        """
        self.epsilon = epsilon
        # Initialize unconstrained matrix W scaled by 1/sqrt(d_state)
        # to ensure variance stability.
        unscaled_W = jax.random.normal(key, (d_state, d_state), dtype=jnp.float32)
        self.W = unscaled_W * (1.0 / math.sqrt(d_state))

    @property
    def Gamma(self) -> Float[Array, "d_state d_state"]:
        """
        Dynamically computes the symmetric positive-definite matrix Γ.
        Must apply tril on the parameter directly here to avoid gradient bleeding
        into the upper triangle.
        """
        # Extract lower triangular part
        L = jnp.tril(self.W)
        
        # Compute L @ L^T
        gamma = L @ L.T
        
        # Add diagonal jitter for numerical stability
        jitter = self.epsilon * jnp.eye(self.W.shape[0], dtype=jnp.float32)
        
        return gamma + jitter

    def __call__(self, x: Float[Array, "d_state"]) -> Float[Array, "d_state"]:
        """
        Computes the matrix-vector product Γx.
        
        Args:
            x: 1D state vector of shape (d_state,).
            
        Returns:
            The product Γx of shape (d_state,).
        """
        return self.Gamma @ x
