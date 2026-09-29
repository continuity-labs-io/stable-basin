Context: Stable Basin repo. Trained engine from benchmark 05 (configs/aging_resilience.yaml).
Three checks on whether the model has the structure the code claims. No training. Use
src.echo.metrics.energy_landscape.ScalarEnergy and graph.forced_unroll.

States: forced-unroll 20 held-out TEST worms (4 windows each, 50-step burn-in), as in
11_null_control. Keep per-worm trajectories.

TEST A: Markov blanket in the learned energy
- Get the micro partition index ranges (internal, sensory, active, external) from
  graph.flow_factor.micro_hull.
- For 500 pooled states, compute H = jax.hessian(ScalarEnergy(graph.ebm))(x) and take the micro
  block H_u = H[:d_micro, :d_micro].
- Per state: r = ||H_u[internal, external]||_F / ||H_u||_F. Report median and 95th percentile.
- Gate: PASS if the 95th percentile of r < 0.01. Otherwise FAIL: the energy couples internal
  and external states directly, so the masks on Q and Gamma do not create a blanket.

TEST B: is the macro level slower than the micro level?
- For each worm, compute the integrated autocorrelation time (sum of the ACF up to its first
  zero crossing) of every micro internal dim and every macro internal dim (macro dims start at
  d_micro; use graph.flow_factor.macro_hull.d_internal).
- Report median tau_macro / median tau_micro with a bootstrap 95% CI over worms.
- Gate: PASS if the lower CI bound > 2.

TEST C: does the macro level predict itself?
- Fit ridge regressions predicting the macro internal state at t+1 from (i) the last 5 macro
  states and (ii) the last 5 full states. Train on 10 worms, test on the other 10.
- Report closure = test MSE(ii) / test MSE(i).
- Gate: PASS if closure >= 0.9 (the micro detail adds little once you know the macro past).

Write output/benchmarks/aging_resilience/structure_checks.json. Report effect sizes and CIs,
no p-values.
