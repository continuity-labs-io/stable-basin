# zero_var_guard_csd_metric
aPlease harden the `calculate_csd` function against baseline noise from dead/quiescent biological channels.

Currently, it computes `var_t` and `ar1_t` across all features using a simple `.mean()`. If a biological channel flatlines (variance approaches zero), `F.cosine_similarity` handles the exact divide-by-zero internally, but the resulting values become highly scaled floating-point noise. This pollutes the global CSD index and causes false-positive instability alerts.

Please update the `for t in range(window_size, time_steps + 1):` loop:
1. Calculate the per-channel variance of `z_win`: `channel_vars = torch.var(z_win, dim=0)`
2. Create a boolean mask identifying active channels: `active_mask = channel_vars > 1e-8`
3. If no channels are active (`~active_mask.any()`), append `0.0` to `csd_scores` for that frame and `continue`.
4. Otherwise, filter `z_win` to only include the active channels (e.g., `z_win_active = z_win[:, active_mask]`) before computing the mean for `var_t` and `ar1_t` (so you are only calculating variance and cosine similarity on active channels).
