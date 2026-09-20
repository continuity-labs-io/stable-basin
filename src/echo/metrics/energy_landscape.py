import jax
import jax.numpy as jnp
from typing import Dict, Callable
import equinox as eqx

@eqx.filter_jit
def calculate_curvature(energy_fn: Callable, x: jax.Array) -> Dict[str, jax.Array]:
    """
    Computes the curvature metrics for a single state vector using a generic energy function.
    
    Args:
        energy_fn: A function that takes a 1D state vector and returns a scalar energy.
        x: A 1D state vector of shape (d_state,)
        
    Returns:
        A dictionary containing:
        - hessian_trace: A scalar quantifying the steepness of the basin.
        - eigenvalues: A 1D array of the Hessian's eigenvalues.
        - hessian_rank: A scalar representing the rank of the Hessian.
        - hessian_nullity: A scalar representing the nullity of the Hessian.
    """
    # 1. Compute Hessian matrix H using forward-over-reverse autodiff
    H = jax.hessian(energy_fn)(x)
    
    # 2. Compute eigenvalues
    # Because the Hessian of a smooth scalar field is strictly symmetric, 
    # we can safely use the highly optimized eigvalsh.
    eigenvalues = jnp.linalg.eigvalsh(H)
    
    # 3. Compute trace (sum of eigenvalues)
    hessian_trace = jnp.sum(eigenvalues)
    
    # 4. Compute Rank and Nullity (Degeneracy / Route Diversity)
    threshold = 1e-4
    hessian_rank = jnp.sum(jnp.abs(eigenvalues) > threshold)
    hessian_nullity = eigenvalues.shape[0] - hessian_rank
    
    return {
        "hessian_trace": hessian_trace,
        "eigenvalues": eigenvalues,
        "hessian_rank": hessian_rank,
        "hessian_nullity": hessian_nullity
    }
    
@eqx.filter_jit
def batch_calculate_curvature(energy_fn: Callable, x_seq: jax.Array) -> Dict[str, jax.Array]:
    """
    Computes the curvature metrics over an unrolled trajectory.
    
    Args:
        energy_fn: A function that takes a 1D state vector and returns a scalar energy.
        x_seq: A 2D array of shape (time_steps, d_state)
        
    Returns:
        A dictionary containing time-series arrays for each metric.
    """
    return jax.vmap(calculate_curvature, in_axes=(None, 0))(energy_fn, x_seq)
