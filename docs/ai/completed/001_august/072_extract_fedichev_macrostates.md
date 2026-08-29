# Extract Fedichev Macrostates

**Context:** We are updating `src/metrics/metrics.py` to support our new "Flight
Recorder" dashboard. This dashboard is based on the Fedichev-Gruber minimal
model of aging, which defines aging through three macroscopic variables: z₀
(fast dynamic stress response), Z (slow cumulative entropic damage), and ε₀ (the
critical recovery rate/bifurcation point).

**Task:** Add a new method named `extract_fedichev_macrostates` to the existing
`ThermodynamicMetrics` class. Do not modify the existing PyDMD or CSD logic; we
are simply orchestrating them to output a clean dictionary of 1D arrays for
plotting.

**Requirements:**

1. **Method Signature:**
   `def extract_fedichev_macrostates(self, z_baseline: torch.Tensor, z_perturbed: torch.Tensor, window_size: int = 4) -> dict:`
2. **Inputs:** `z_baseline` and `z_perturbed` are tensors of shape
   `[Time, Embed_Dim]`.
3. **Variable Z (Entropic Damage):** Calculate the Euclidean distance between
   `z_baseline` and `z_perturbed` at each time step (this is the
   `path_divergence` from existing hysteresis logic). Calculate `Z_t` as the
   cumulative trapezoidal integration (using `torch.cumulative_trapezoid` or
   `scipy.integrate.cumulative_trapezoid`) of this divergence over time.
4. **Variable z₀_volatility (Dynamic Response):** Call
   `self.calculate_csd(z_perturbed, window_size)` to get the fast kinetic
   volatility.
5. **Variable ε₀ (Criticality):** Call
   `self.calculate_ksm(z_perturbed, window_size)` to get the system's dynamic
   stability.
6. **Return Format:** Return a dictionary containing the keys
   `"Z_entropic_damage"`, `"z0_volatility"`, and `"epsilon_0_ksm"`, where each
   value is a flat Python list or numpy array of length `Time`.
7. **Code Quality:** Ensure it handles mismatched sequence lengths gracefully by
   truncating to the minimum length of `z_baseline` and `z_perturbed` before
   calculation, matching the style of the existing `calculate_hysteresis`
   method.
