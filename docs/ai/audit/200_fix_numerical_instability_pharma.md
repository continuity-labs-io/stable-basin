We are experiencing a numerical instability issue during `make
worm-gait-pharmacology`. The Euler-Maruyama integration blows up (returning
`NaN`s) because the 4PL curve fit extrapolates an EC50 dose of ~30.0, which
creates restorative gradients too steep for our `dt = 0.01` timestep.

Instead of reducing the integration timestep (which would significantly slow
down the pipeline), please implement a two-part fix to make the physics engine
stable and prevent unrealistic dose extrapolation:

1. **Cap the Extrapolated EC50 Bound
   (`src/benchmarks/worm_gait/09_pharmacological_translation.py`)** The `bounds`
   tuple provided to `curve_fit` currently allows the `ec50` to extrapolate up
   to `max(lambdas) * 10.0`. Please change this upper bound from `max(lambdas) *
   10.0` to `max(lambdas)` to constrain the fit strictly within our tested
   clinical sweep range.

2. **Add Drift Clipping in the Physics Unroller
   (`src/benchmarks/worm_gait/core.py`)** In the `simulate_sde` function, the
   `scan_step` closure computes the `drift` vector but does not bound it. This
   differs from our main `Thermostat` module (`src/echo/physics/thermostat.py`),
   which utilizes clipping to prevent gradient explosion. Please add `drift =
   jnp.clip(drift, -100.0, 100.0)` immediately after `drift` is calculated and
   before the Euler update step. This will prevent the gradients from exploding
   to infinity in a single discrete step when lambda is high.

Please implement these two changes. They will elegantly resolve the `NaN`
propagation without penalizing the simulation runtime!
