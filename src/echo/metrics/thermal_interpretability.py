import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict

from src.echo.primitives.ebm import PrecisionWeightedEBM

class HessianCurvatureTracker(eqx.Module):
    """
    Extracts the topographical geometry of the biological attractor basin 
    at runtime by analyzing the Hessian curvature of the Energy landscape.
    """
    ebm: eqx.Module

    def __init__(self, ebm: eqx.Module):
        """
        Initializes the tracker with an Energy-Based Model.
        
        Args:
            ebm: An instance of PrecisionWeightedEBM (or structurally similar).
        """
        self.ebm = ebm

    @eqx.filter_jit
    def calculate_curvature(self, x: jax.Array) -> Dict[str, jax.Array]:
        """
        Computes the curvature metrics for a single state vector.
        
        Args:
            x: A 1D state vector of shape (d_state,)
            
        Returns:
            A dictionary containing:
            - hessian_trace: A scalar quantifying the steepness of the basin.
            - explicit_precision_trace: A scalar trace of the network's Precision output.
            - eigenvalues: A 1D array of the Hessian's eigenvalues.
        """
        def energy_fn(state):
            # Evaluate EBM and return ONLY the scalar energy
            e, _ = self.ebm(state)
            return e
            
        # 1. Compute Hessian matrix H using forward-over-reverse autodiff
        H = jax.hessian(energy_fn)(x)
        
        # 2. Compute eigenvalues
        # Because the Hessian of a smooth scalar field is strictly symmetric, 
        # we can safely use the highly optimized eigvalsh.
        eigenvalues = jnp.linalg.eigvalsh(H)
        
        # 3. Compute trace (sum of eigenvalues)
        hessian_trace = jnp.sum(eigenvalues)
        
        # 4. Evaluate EBM normally to get explicit Precision matrix Pi
        _, Pi = self.ebm(x)
        explicit_precision_trace = jnp.trace(Pi)
        
        # 5. Compute Rank and Nullity (Degeneracy / Route Diversity)
        threshold = 1e-4
        hessian_rank = jnp.sum(jnp.abs(eigenvalues) > threshold)
        hessian_nullity = eigenvalues.shape[0] - hessian_rank
        
        return {
            "hessian_trace": hessian_trace,
            "explicit_precision_trace": explicit_precision_trace,
            "eigenvalues": eigenvalues,
            "hessian_rank": hessian_rank,
            "hessian_nullity": hessian_nullity
        }
        
    @eqx.filter_jit
    def batch_calculate_curvature(self, x_seq: jax.Array) -> Dict[str, jax.Array]:
        """
        Computes the curvature metrics over an unrolled trajectory.
        
        Args:
            x_seq: A 2D array of shape (time_steps, d_state)
            
        Returns:
            A dictionary containing time-series arrays for each metric.
        """
        return jax.vmap(HessianCurvatureTracker.calculate_curvature, in_axes=(None, 0))(self, x_seq)
