# Plan: Enforce Strict Jaxtyping Signatures

## Overview
As per Audit Fix 5, several files use generic `jax.Array` or `torch.Tensor` types or lack return types entirely, defeating the purpose of `jaxtyping`. This plan outlines the specific shape annotations to be added to ensure strict runtime type-checking with `beartype` and `jaxtyping`.

## Files to Modify

### 1. `src/echo/primitives/thermalizer.py`
- Import `Float` and `Array` from `jaxtyping`.
- **`TorxThermalizer.__call__`**:
  - Update `x_init: jax.Array` to `x_init: Float[Array, "d_state"]`.
  - Update return type from `jax.Array` to `Float[Array, "n_steps d_state"]` (as it returns a trajectory).
- **`ForcedTorxThermalizer.__call__`**:
  - Update `x_init: jax.Array` to `x_init: Float[Array, "d_state"]`.
  - Update `seq: jax.Array | None` to `seq: Float[Array, "seq_len d_seq"] | None`.
  - Update `omega_seq: jax.Array | None` to `omega_seq: Float[Array, "seq_len d_omega"] | None`.
  - Update `q_mask: jax.Array | None` to `q_mask: Float[Array, "d_state"] | None`.
  - Update return type from `jax.Array` to `Float[Array, "seq_len d_state"]`.

### 2. `src/echo/architecture/predictive_coding_graph.py`
- Import `Float` and `Array` from `jaxtyping`.
- **`JointEBM.__call__`**:
  - Add type hint for `x`: `x: Float[Array, "d_state"]`.
  - Add return type: `tuple[Float[Array, ""], Float[Array, "d_state d_state"]]` (Energy scalar and precision matrix).
- **`PredictiveCodingGraph.__call__`**:
  - Update `x_init: jax.Array` to `x_init: Float[Array, "d_state"]`.
  - Update return type from `jax.Array` to `Float[Array, "n_steps d_state"]`.
- **`PredictiveCodingGraph.forced_unroll`**:
  - Add `@jaxtyped(typechecker=beartype)` decorator.
  - Update `x_init: jax.Array` to `x_init: Float[Array, "d_state"]`.
  - Update `seq: jax.Array | None` to `seq: Float[Array, "seq_len d_seq"] | None`.
  - Update `omega_seq: jax.Array | None` to `omega_seq: Float[Array, "seq_len d_omega"] | None`.
  - Update `q_mask: jax.Array | None` to `q_mask: Float[Array, "d_state"] | None`.
  - Update return type from `jax.Array` to `Float[Array, "seq_len d_state"]`.

### 3. `src/harness/sensor_fusion_predictor.py`
- Import `Float` and `Tensor` from `jaxtyping`.
- **`SensorFusionPredictor.forward`**:
  - Update `x_raw: torch.Tensor` to `x_raw: Float[Tensor, "batch seq dim"]`.
  - Update `mask: Optional[torch.Tensor] = None` to `mask: Float[Tensor, "batch seq dim"] | None = None`.
  - Update return type to `tuple[Float[Tensor, "batch out_dim"], Float[Tensor, "batch seq d_model"], Float[Tensor, "..."] | None]`. (Since `reconstructed_t` is initialized to None and returned, it might just be `None` or a tensor).
- **`SensorFusionPredictor.get_hidden_states`**:
  - Add `@jaxtyped(typechecker=beartype)` decorator.
  - Add type hints: `x: Float[Tensor, "batch seq dim"]`, `mask: Float[Tensor, "batch seq dim"] | None = None`.
  - Add return type: `Float[Tensor, "batch seq d_model"]`.

### 4. `src/models/encoders/topo_encoder.py`
- **`TopoEncoder.forward`**:
  - Input `x: Float[Tensor, "batch time 2 64 64"]` is already correct.
  - Add missing return type. Using Python `typing.Union` or `|` syntax: `Float[Tensor, "batch d_model"] | tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch time d_model"]]`.

## Execution Steps
1. Upon user approval, this plan will be copied to `docs/ai/audit/227_enforce_strict_jaxtyping.md` to persist the strategy.
2. The specified edits will be applied to the four files.
3. Tests will be executed to ensure all typechecking passes at runtime.
4. Changes will be committed.
