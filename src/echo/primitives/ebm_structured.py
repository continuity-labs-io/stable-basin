import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, jaxtyped
from beartype import beartype
from typing import Tuple

class StructuredPrecisionEBM(eqx.Module):
    mlp: eqx.nn.MLP
    energy_head: eqx.nn.Linear
    diag_head: eqx.nn.Linear
    low_rank_head: eqx.nn.Linear
    d_state: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        d_state: int,
        hidden_size: int,
        depth: int,
        key: PRNGKeyArray,
        rank: int = 4,
        epsilon: float = 1e-4,
    ):
        self.d_state = d_state
        self.rank = rank
        self.epsilon = epsilon

        key_mlp, key_energy, key_diag, key_low_rank = jax.random.split(key, 4)

        self.mlp = eqx.nn.MLP(
            in_size=d_state,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=key_mlp,
        )
        self.energy_head = eqx.nn.Linear(hidden_size, 1, key=key_energy)
        self.diag_head = eqx.nn.Linear(hidden_size, d_state, key=key_diag)
        self.low_rank_head = eqx.nn.Linear(hidden_size, d_state * rank, key=key_low_rank)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "d_state"]
    ) -> Tuple[Float[Array, ""], Float[Array, "d_state d_state"]]:
        h = self.mlp(x)
        
        energy_raw = self.energy_head(h)
        e_mlp = jnp.squeeze(jax.nn.softplus(energy_raw))
        e_prior = 0.5 * 0.001 * jnp.sum(x**2)
        energy = e_prior + e_mlp

        v_diag = jax.nn.softplus(self.diag_head(h)) + self.epsilon
        U = self.low_rank_head(h).reshape((self.d_state, self.rank))
        
        precision = jnp.diag(v_diag) + U @ U.T

        return energy, precision
