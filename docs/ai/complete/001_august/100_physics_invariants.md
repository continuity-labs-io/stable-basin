Act as a Staff Machine Learning Engineer. We need to build a pure PyTorch
invariant test suite for the `ThermodynamicMetrics` class to guarantee our math
is sound before we train the Mamba models.

Write a pytest script in `tests/metrics/test_physics_invariants.py`.

1. Instantiate ThermodynamicMetrics.
2. Generate a perfectly stable 10D biological sine wave (Time=500). Assert that
   the calculate_ksm function returns values consistently > 0.95.
3. Generate a 10D tensor of pure Gaussian white noise. Assert that the
   calculate_ksm function returns values < 0.2.
4. Keep the code simple, fast, and completely isolated from the ML models. Do
   not use any external dependencies outside of torch, numpy, and pydmd.
5. Ensure the tests pass.
