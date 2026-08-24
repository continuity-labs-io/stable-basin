"""
Solenoidal Flow Module for Project Echo.

This module provides the energy-conserving rotational dynamics (Q) of a biological 
attractor basin. The flow is strictly skew-symmetric, enforcing the thermodynamic 
invariant that it performs no work on the system (x^T Q x = 0).
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray

class SolenoidalFlow(eqx.Module):
    """
    Equinox module representing solenoidal (skew-symmetric) flow.
    """
    
    W: Float[Array, "d_state d_state"]
    
    def __init__(self, d_state: int, key: PRNGKeyArray):
        """
        Initializes the unconstrained weight matrix W.
        
        Args:
            d_state: Dimensionality of the state vector.
            key: JAX PRNG key for initialization.
        """
        # Scale by 1.0 / sqrt(d_state) for variance stability
        scale = 1.0 / math.sqrt(d_state)
        self.W = jax.random.normal(key, (d_state, d_state)) * scale
        
    @property
    def Q(self) -> Float[Array, "d_state d_state"]:
        """
        Dynamically computes and returns the skew-symmetric matrix Q.
        
        Returns:
            Q = W - W^T
        """
        return self.W - self.W.T
        
    def __call__(self, x: Float[Array, "d_state"]) -> Float[Array, "d_state"]:
        """
        Computes the matrix-vector product Qx.
        
        Args:
            x: A 1D state vector of shape (d_state,).
            
        Returns:
            The resulting flow vector.
        """
        return self.Q @ x
