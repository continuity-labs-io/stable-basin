import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from typing import Dict

class MarkovHull(eqx.Module):
    """
    Mathematically enforces a Markov Blanket partition on a biological state tensor.
    Partitions the universe state into internal (μ), sensory (s), active (a),
    and external (η) components.
    Enforces the fundamental law: internal state must not interact directly with 
    the external state.
    
    Attributes:
        d_internal: Dimensionality of the internal state (μ).
        d_sensory: Dimensionality of the sensory state (s).
        d_active: Dimensionality of the active state (a).
        d_external: Dimensionality of the external state (η).
        d_state: Total dimensionality of the universe state (μ + s + a + η).
        D_s: Optional Sensory Degradation Matrix to model sensory blindness.
    """
    d_internal: int = eqx.field(static=True)
    d_sensory: int = eqx.field(static=True)
    d_active: int = eqx.field(static=True)
    d_external: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    D_s: jax.Array | None = eqx.field(default=None)

    def __init__(
        self,
        d_internal: int,
        d_sensory: int,
        d_active: int,
        d_external: int,
        D_s: jax.Array | None = None
    ):
        """
        Initializes the MarkovHull with component dimensions.
        """
        self.d_internal = d_internal
        self.d_sensory = d_sensory
        self.d_active = d_active
        self.d_external = d_external
        self.d_state = d_internal + d_sensory + d_active + d_external
        self.D_s = D_s

    def partition(self, x: Float[Array, "d_state"]) -> Dict[str, Float[Array, "..."]]:
        """
        Splits a flat 1D state array into its functional components.
        
        Args:
            x: Flat 1D JAX array of size d_state.
            
        Returns:
            Dictionary containing 'internal', 'sensory', 'active', 'external' slices.
        """
        idx_s = self.d_internal
        idx_a = idx_s + self.d_sensory
        idx_e = idx_a + self.d_active
        
        return {
            "internal": x[:idx_s],
            "sensory": x[idx_s:idx_a],
            "active": x[idx_a:idx_e],
            "external": x[idx_e:]
        }

    def reconstruct(self, partitions: Dict[str, Float[Array, "..."]]) -> Float[Array, "d_state"]:
        """
        Concatenates functional components back into a flat 1D array.
        
        Args:
            partitions: Dictionary with 'internal', 'sensory', 'active', 'external' arrays.
            
        Returns:
            Flat 1D JAX array of size d_state.
        """
        return jnp.concatenate([
            partitions["internal"],
            partitions["sensory"],
            partitions["active"],
            partitions["external"]
        ])

    def apply_sensory_degradation(self, x: jax.Array) -> jax.Array:
        """
        Degrades the sensory partition of the state if a sensory degradation matrix (D_s) is provided.
        """
        if self.D_s is None:
            return x
            
        partitions = self.partition(x)
        s_obs = self.D_s @ partitions["sensory"]
        partitions["sensory"] = s_obs
        return self.reconstruct(partitions)

    def get_topology_mask(self) -> Float[Array, "d_state d_state"]:
        """
        Constructs a binary adjacency mask enforcing the Markov Blanket.
        
        The internal states and external states are topologically severed (0.0).
        All other connections (including the blanket and self-interactions) are permitted (1.0).
        
        Returns:
            A (d_state, d_state) binary mask matrix.
        """
        # Start with fully connected (all ones)
        mask = jnp.ones((self.d_state, self.d_state), dtype=jnp.float32)
        
        idx_s = self.d_internal
        idx_e = self.d_internal + self.d_sensory + self.d_active
        
        # internal -> external block severed
        mask = mask.at[:idx_s, idx_e:].set(0.0)
        # external -> internal block severed
        mask = mask.at[idx_e:, :idx_s].set(0.0)
        
        return mask
