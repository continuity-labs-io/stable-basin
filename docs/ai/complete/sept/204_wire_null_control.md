Reviewer 2 has provided two new files (`11_null_control.py` and `synthetic_aging.py`) to serve as our ground-truth synthetic positive control. We need to wire these into our pipeline.

1. **Update `src/data/behavior/celegans_gait_dataset.py`:**
   Rename the `is_aged` parameter to `inject_synthetic_degradation` everywhere in the codebase (including all instantiations in scripts 01, 02, 03, 05, 06, and 10). 
   In `RealEigenwormDataset.__init__`, delete the old noise injection (`traj = traj * 0.5 + torch.randn_like(traj) * 0.2`). Instead, import `slow_amplitude_relaxation` from `src.data.behavior.synthetic_aging`. If `inject_synthetic_degradation` is True, apply `traj = torch.tensor(slow_amplitude_relaxation(traj.numpy(), slowdown=3.0, pair=(0, 1)), dtype=torch.float32)`.

2. **Update the Makefile:**
   Add a new target `worm-gait-null-control` that runs `python -m src.benchmarks.worm_gait.11_null_control --config configs/worm_gait_ebm.yaml`. Add this to the `worm-gait-experiments` execution chain.

3. **Update Plot Labels:**
   In `05_worm_gait_aging_ebm.py`, `07_worm_gait_intervention.py`, and `10_animate_worm_gait.py`, replace all legend strings and plot titles referencing "Young (Day 1-3)" and "Old (Day 9+)" with "Clean Baseline" and "Synthetically Degraded".
