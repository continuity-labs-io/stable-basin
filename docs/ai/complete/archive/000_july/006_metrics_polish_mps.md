Context: We are finalizing the Tier 1 Sandbox for Project MELD. The initial E2E
script ran successfully, but the thermodynamic metrics (KSM) suffer from an
eigenvalue inversion on stable frames, the Hysteresis rollout is too short, and
the script defaults to CPU on Macs instead of utilizing Apple Silicon (MPS).

Task 1: Update `src/metrics/thermodynamics.py` Fix the rank-deficiency and
eigenvalue logic in the KSM calculation.

- In `calculate_ksm(self, z_sequence, window_size=4)`:
  - Add Tikhonov regularization to the `S_inv` calculation to prevent
    divide-by-zero on highly stable frames:
    `S_inv = torch.diag(S / (S**2 + 1e-4))`
  - Fix the `ksm` calculation so that an eigenvalue of 1.0 results in a KSM of
    1.0 (Healthy), and diverging eigenvalues drop the KSM toward 0.0 (Crash):
    ```python
    # KSM is bounded [0, 1]. A stable system has max_eig near 1.0.
    ksm = 1.0 / (1.0 + abs(max_eig - 1.0))
    ksm_scores.append(ksm)
    ```

Task 2: Update `src/demo/e2e_demo.py` Fix the hardware device mapping and expand
the Hysteresis rollout.

- Replace the `device = ...` line inside `main()` with:
  ```python
  if torch.cuda.is_available():
      device = torch.device("cuda")
  elif torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  ```
- In the "Simulating Biological Rescue" section:
  - Increase the rollout loop from 2 to 8 steps: `for _ in range(8):`
  - To calculate hysteresis over this extended rollout, we need the healthy path
    to match the length. Replace `z_healthy_path = z_fused_healthy[7:10, :]`
    with:
    ```python
    z_healthy_base = z_fused_healthy[7:, :]
    pad_len = z_rescue_path.shape[0] - z_healthy_base.shape[0]
    if pad_len > 0:
        padding = z_healthy_base[-1:, :].repeat(pad_len, 1)
        z_healthy_path = torch.cat([z_healthy_base, padding], dim=0)
    else:
        z_healthy_path = z_healthy_base[: z_rescue_path.shape[0], :]
    ```
