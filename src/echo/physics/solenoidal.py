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
from jaxtyping import Float, Array, PRNGKeyArray, jaxtyped
from beartype import beartype

from typing import Any

class SolenoidalFlow(eqx.Module):
    """
    Equinox module representing solenoidal (skew-symmetric) flow.
    """
    
    W: Float[Array, "d_state d_state"]
    hull: Any
    
    @jaxtyped(typechecker=beartype)
    def __init__(self, d_state: int, key: PRNGKeyArray, hull: Any = None):
        """
        Initializes the unconstrained weight matrix W.
        
        Args:
            d_state: Dimensionality of the state vector.
            key: JAX PRNG key for initialization.
            hull: Optional MarkovHull to enforce topological constraints.
        """
        if not isinstance(d_state, int) or d_state <= 0:
            raise ValueError(f"d_state must be a strictly positive integer, got {d_state}")
            
        if hull is not None:
            if not hasattr(hull, 'd_state') or hull.d_state != d_state:
                raise ValueError(f"hull.d_state ({getattr(hull, 'd_state', None)}) must exactly match d_state ({d_state})")
            if not all(hasattr(hull, attr) for attr in ('d_internal', 'd_sensory', 'd_active')):
                raise TypeError("hull must possess d_internal, d_sensory, and d_active properties.")
                
        self.hull = hull
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
        Q_unconstrained = self.W - self.W.T
        if self.hull is not None:
            return Q_unconstrained * self.hull.get_topology_mask()
        return Q_unconstrained
        
    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "d_state"]) -> Float[Array, "d_state"]:
        """
        Computes the matrix-vector product Qx.
        
        Args:
            x: A 1D state vector of shape (d_state,).
            
        Returns:
            The resulting flow vector.
        """
        return self.Q @ x
