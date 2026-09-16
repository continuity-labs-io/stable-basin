Please update the `ThermodynamicMetrics` class to include a new method called
`calculate_lle(self, z_sequence, window_size=4, dt=1.0)`. This method will
compute the Local Lyapunov Exponent (LLE) over a sliding window to measure the
stability of the biological attractor basin.

To ensure architectural simplicity and speed on edge GPUs, reuse the exact
Truncated SVD and Dynamic Mode Decomposition (DMD) logic currently found in the
`calculate_ksm` method. Use this logic to extract the local linear operator
`A_tilde` and find `max_eig`, which is the maximum absolute eigenvalue of the
system.

Instead of bounding the output using an exponential envelope like KSM, calculate
the LLE using the formula: LLE = ln(max_eig) / dt. Use
`math.log(max_eig + 1e-7) / dt` to safely compute the natural logarithm while
avoiding log(0) errors.

Return a list of LLE scores, ensuring you pad the initial frames with 0.0 to
match the exact length of `time_steps`.
