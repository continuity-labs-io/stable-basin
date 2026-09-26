I am implementing "Roadmap Level 2: Breaking the Dimensionality Wall". First, we need to create an O(d*r) parameter alternative to the dense precision matrix in the EBM.

1. Open `src/echo/primitives/ebm_structured.py` (create it).
2. Add necessary imports: `jax`, `jax.numpy as jnp`, `equinox as eqx`, `jaxtyping` imports (`Float`, `Array`, `PRNGKeyArray`, `jaxtyped`), `beartype`, and `typing.Tuple`.
3. Define `class StructuredPrecisionEBM(eqx.Module):`
4. This class should mimic `PrecisionWeightedEBM` but predict a structured precision matrix: $\Pi(x) = \text{diag}(\mathbf{v}(x)) + \mathbf{U}(x) \mathbf{U}(x)^T$.
5. Add attributes: `mlp: eqx.nn.MLP`, `energy_head: eqx.nn.Linear`, `diag_head: eqx.nn.Linear`, `low_rank_head: eqx.nn.Linear`, `d_state: int = eqx.field(static=True)`, `rank: int = eqx.field(static=True)`, `epsilon: float = eqx.field(static=True)`.
6. The constructor `__init__(self, d_state: int, hidden_size: int, depth: int, key: PRNGKeyArray, rank: int = 4, epsilon: float = 1e-4)` should initialize:
   - `self.d_state = d_state`, `self.rank = rank`, `self.epsilon = epsilon`.
   - Split `key` into 4 keys (`key_mlp`, `key_energy`, `key_diag`, `key_low_rank`).
   - `self.mlp` as `eqx.nn.MLP(in_size=d_state, out_size=hidden_size, width_size=hidden_size, depth=depth, activation=jax.nn.gelu, key=key_mlp)`.
   - `self.energy_head` as `eqx.nn.Linear(hidden_size, 1, key=key_energy)`.
   - `self.diag_head` as `eqx.nn.Linear(hidden_size, d_state, key=key_diag)`.
   - `self.low_rank_head` as `eqx.nn.Linear(hidden_size, d_state * rank, key=key_low_rank)`.
7. The `__call__(self, x: Float[Array, "d_state"]) -> Tuple[Float[Array, ""], Float[Array, "d_state d_state"]]` method should:
   - Run `h = self.mlp(x)`.
   - Compute `energy`: `energy_raw = self.energy_head(h)`, `e_mlp = jnp.squeeze(jax.nn.softplus(energy_raw))`, `e_prior = 0.5 * 0.001 * jnp.sum(x**2)`, `energy = e_prior + e_mlp`.
   - Compute the diagonal: `v_diag = jax.nn.softplus(self.diag_head(h)) + self.epsilon`.
   - Compute the low-rank component: `U = self.low_rank_head(h).reshape((self.d_state, self.rank))`.
   - Construct and return the dense representation (we materialize it here for backward compatibility with the current ODE unroller, but the parameter footprint is drastically reduced): `precision = jnp.diag(v_diag) + U @ U.T`.
   - Return `energy, precision`.
