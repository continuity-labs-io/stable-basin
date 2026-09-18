import math
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, jaxtyped
from beartype import beartype

from typing import Any

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
    hull: Any

    @jaxtyped(typechecker=beartype)
    def __init__(self, d_state: int, key: PRNGKeyArray, epsilon: float = 1e-4, hull: Any = None):
        """
        Initializes the DissipativeFriction module.
        
        Args:
            d_state: The dimensionality of the state vector.
            key: PRNG key for initialization.
            epsilon: Jitter added to the diagonal to ensure strict positive-definiteness.
            hull: Optional MarkovHull to enforce topological constraints.
        """
        if not isinstance(d_state, int) or d_state <= 0:
            raise ValueError(f"d_state must be a strictly positive integer, got {d_state}")
        if not isinstance(epsilon, (float, int)) or epsilon < 0:
            raise ValueError(f"epsilon must be a non-negative float, got {epsilon}")
            
        if hull is not None:
            if not hasattr(hull, 'd_state') or hull.d_state != d_state:
                raise ValueError(f"hull.d_state ({getattr(hull, 'd_state', None)}) must exactly match d_state ({d_state})")
            if not all(hasattr(hull, attr) for attr in ('d_internal', 'd_sensory', 'd_active')):
                raise TypeError("hull must possess d_internal, d_sensory, and d_active properties.")
        self.epsilon = epsilon
        self.hull = hull
        # Initialize unconstrained matrix W scaled by 1/sqrt(d_state)
        # to ensure variance stability.
        unscaled_W = jax.random.normal(key, (d_state, d_state), dtype=jnp.float32)
        self.W = unscaled_W * (1.0 / math.sqrt(d_state))

    @property
    def L(self) -> Float[Array, "d_state d_state"]:
        """
        Returns the topologically constrained lower-triangular Cholesky factor.
        """
        L_orig = jnp.tril(self.W)
        if self.hull is not None:
            # Enforce Markov Blanket zeroing on the external-internal (bottom-left) block
            idx_s = self.hull.d_internal
            idx_e = self.hull.d_internal + self.hull.d_sensory + self.hull.d_active
            # L is lower triangular, so L_ie (external rows, internal cols) must be zeroed to preserve PSD.
            return L_orig.at[idx_e:, :idx_s].set(0.0)
        return L_orig

    @property
    def Gamma(self) -> Float[Array, "d_state d_state"]:
        """
        Dynamically computes the symmetric positive-definite matrix Γ.
        Must apply tril on the parameter directly here to avoid gradient bleeding
        into the upper triangle.
        """
        # Extract constrained lower triangular part
        L = self.L
        
        # Compute L @ L^T
        gamma = L @ L.T
        
        # Add diagonal jitter for numerical stability
        jitter = self.epsilon * jnp.eye(self.W.shape[0], dtype=jnp.float32)
        
        return gamma + jitter

    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "d_state"]) -> Float[Array, "d_state"]:
        """
        Computes the matrix-vector product Γx.
        
        Args:
            x: 1D state vector of shape (d_state,).
            
        Returns:
            The product Γx of shape (d_state,).
        """
        return self.Gamma @ x
