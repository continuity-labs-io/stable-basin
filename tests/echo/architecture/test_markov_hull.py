import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from src.echo.architecture.markov_hull import MarkovHull

def test_markov_hull_reversibility():
    """
    Test that reconstruct(partition(x)) perfectly matches the original array x.
    """
    # ARRANGE
    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 4
    total_dims = 10
    key = jax.random.PRNGKey(42)
    
    hull = MarkovHull(d_internal, d_sensory, d_active, d_external)
    x = jax.random.normal(key, (total_dims,), dtype=jnp.float32)
    
    # ACT
    partitions = hull.partition(x)
    x_reconstructed = hull.reconstruct(partitions)
    
    # ASSERT
    assert jnp.allclose(x, x_reconstructed), "Reconstructed array does not match original."

def test_markov_hull_partition_shapes():
    """
    Test that partition(x) correctly splits the array into the expected shapes.
    """
    # ARRANGE
    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 4
    total_dims = 10
    key = jax.random.PRNGKey(123)
    
    hull = MarkovHull(d_internal, d_sensory, d_active, d_external)
    x = jax.random.normal(key, (total_dims,), dtype=jnp.float32)
    
    # ACT
    partitions = hull.partition(x)
    
    # ASSERT
    assert partitions["internal"].shape == (d_internal,), "Internal partition shape mismatch."
    assert partitions["sensory"].shape == (d_sensory,), "Sensory partition shape mismatch."
    assert partitions["active"].shape == (d_active,), "Active partition shape mismatch."
    assert partitions["external"].shape == (d_external,), "External partition shape mismatch."

def test_markov_hull_topological_mask():
    """
    Test the Causal Sever: ensures the topology mask correctly zeroes out
    internal-external interactions while keeping all others intact.
    """
    # ARRANGE
    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 4
    total_dims = 10
    
    hull = MarkovHull(d_internal, d_sensory, d_active, d_external)
    
    # ACT
    mask = hull.get_topology_mask()
    
    # ASSERT
    assert mask.shape == (total_dims, total_dims), "Mask shape mismatch."
    
    # Top-right block (internal rows 0:2, external cols 6:10) is exactly all zeros
    assert jnp.all(mask[0:2, 6:10] == 0.0), "Top-right internal-external block not completely severed."
    
    # Bottom-left block (external rows 6:10, internal cols 0:2) is exactly all zeros
    assert jnp.all(mask[6:10, 0:2] == 0.0), "Bottom-left external-internal block not completely severed."
    
    # All other blocks should be exactly 1.0. We can check by summing the elements 
    # and ensuring the sum matches (total_dims * total_dims) - 2 * (d_internal * d_external).
    expected_ones = (total_dims * total_dims) - (2 * d_internal * d_external)
    assert jnp.sum(mask) == expected_ones, "Mask has unexpected zero values in permitted regions."
    
    # Explicitly check sensory/active block (2:6, 2:6)
    assert jnp.all(mask[2:6, 2:6] == 1.0), "Blanket self-interaction not fully connected."

def test_markov_hull_jit():
    """
    Asserts that the module and its methods can be passed through jax.jit seamlessly.
    """
    # ARRANGE
    d_internal = 2
    d_sensory = 2
    d_active = 2
    d_external = 4
    total_dims = 10
    key = jax.random.PRNGKey(999)
    
    hull = MarkovHull(d_internal, d_sensory, d_active, d_external)
    x = jax.random.normal(key, (total_dims,), dtype=jnp.float32)
    
    # ACT
    @eqx.filter_jit
    def jit_pipeline(mod, state):
        parts = mod.partition(state)
        # Just mutate it harmlessly to prove JIT works cleanly
        parts["internal"] = parts["internal"] * 2.0
        new_state = mod.reconstruct(parts)
        mask = mod.get_topology_mask()
        return new_state, mask

    x_new, mask = jit_pipeline(hull, x)
    
    # ASSERT
    assert x_new.shape == (total_dims,)
    assert mask.shape == (total_dims, total_dims)
    assert not jnp.any(jnp.isnan(x_new))
    assert not jnp.any(jnp.isnan(mask))
