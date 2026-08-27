ROLE: You are an elite Scientific Machine Learning Engineer specializing in Computational Physics and JAX/PyTorch Data Engineering.

TASK: We are building `src/echo/data/toy/muller_brown.py`. Implement a robust JAX physics simulator for the Müller-Brown potential and wrap it in a PyTorch-compatible `Dataset` called `MullerBrownDataset`. This will generate continuous-time trajectories of a particle moving through the potential field using overdamped Langevin dynamics.

MATHEMATICAL CONSTRAINTS:
1. The Müller-Brown potential V(x, y) is defined as a sum of four exponentials:
   V(x, y) = sum_{i=1}^4 A_i * exp( a_i*(x - x0_i)^2 + b_i*(x - x0_i)*(y - y0_i) + c_i*(y - y0_i)^2 )
   Use the exact standard coefficients:
   A = [-200.0, -100.0, -170.0, 15.0]
   a = [-1.0, -1.0, -6.5, 0.7]
   b = [0.0, 0.0, 11.0, 0.6]
   c = [-10.0, -10.0, -6.5, 0.7]
   x0 = [1.0, 0.0, -0.5, -1.0]
   y0 = [0.0, 0.5, 1.5, 1.0]
2. The dataset must generate continuous trajectories of a particle navigating this 2D surface.
3. Use the Euler-Maruyama method to integrate the overdamped Langevin equation:
   dx = -∇V(x, y) * dt + sqrt(2 * kT * dt) * dW
   (where ∇V is the analytical gradient of the potential, kT is the thermal energy/temperature, dt is the time step, and dW is standard Gaussian noise).

TECHNICAL REQUIREMENTS:
- The physics simulation MUST be written in pure JAX for speed (`jax.numpy`, `jax.grad`, `jax.lax.scan`).
- Define a pure JAX function `muller_brown_potential(state: jax.Array) -> jax.Array` that takes a shape `(2,)` array and returns a scalar.
- Define a JIT-compiled JAX function `generate_trajectory(key, state_init, n_steps, dt, kT)` that unrolls the Langevin dynamics using `jax.lax.scan` and `jax.grad(muller_brown_potential)`. To prevent explosions from steep gradients, apply gradient clipping (`jnp.clip(grad, -100.0, 100.0)`).
- The class `MullerBrownDataset` MUST inherit from `torch.utils.data.Dataset`.
- The `__init__` method must accept: `size: int`, `seq_len: int`, `dt: float = 0.001`, `kT: float = 15.0`, `seed: int = 42`.
- In `__init__`, pre-generate the dataset by using `jax.vmap` over `generate_trajectory` to rapidly generate `size` sequences. Convert the final JAX array to a PyTorch float32 tensor via `torch.from_numpy(np.array(...))`.
- Initialization: Spawn the particles near one of the three known minima (roughly `(-0.5, 1.5)`, `(0.0, 0.5)`, or `(0.5, 0.0)`) by sampling from these centers and adding small Gaussian noise.
- Output Format: `__getitem__` must return a dictionary compatible with the `stable-basin` harness:
  - `"x_raw"`: Shape `(seq_len, 2)`. The generated trajectory [x, y].
  - `"mask"`: Shape `(seq_len, 2)`. A tensor of ones (1.0).
  - `"y_true"`: Shape `(seq_len, 2)`. Identical to `x_raw` (target for forecasting).

TESTING:
Create `tests/echo/data/test_muller_brown.py`. Write a rigorous `pytest` suite that:
1. Shape & Interoperability Test: Instantiates `MullerBrownDataset` with `size=10, seq_len=100`. Asserts `__len__` is 10. Asserts `dataset[0]["x_raw"]` is a PyTorch tensor of shape `(100, 2)` containing no NaNs.
2. The Minimum Test: Evaluates `jax.grad(muller_brown_potential)` at `[-0.558, 1.441]` (the approximate global minimum). Asserts that the gradient magnitude is near zero (`< 2.0`), proving the math correctly identifies the basin floor.
3. Fast JAX Execution: Asserts that `generate_trajectory` can be executed under `jax.jit` without concretization errors.

Write production-grade code with clean docstrings. Focus strictly on this module.
