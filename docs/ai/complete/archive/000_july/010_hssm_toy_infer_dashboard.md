In `src/demo/hssm_toy_model.py`, add the evaluation and plotting logic (The KSM
Dashboard).

1. Write an `evaluate_and_plot(compressor, mamba_engine, device)` function.
2. Generate validation data (batch size 1) for all 3 scenarios: 'homeostasis',
   'corrosion', and 'toxic_shock'.
3. In `torch.no_grad()`, pass each through the compressor and Mamba engine.
   Extract the `frame_distances` (Cosine Distance / Surprise metric).
4. Create a 3-panel matplotlib figure (figsize=(10, 12), save to
   `output/hssm_veto_proof.png`):
   - Top Panel (The Drowning Signal): Plot a 4000-step slice (0.2 seconds) of
     the raw 20kHz GEVI 'homeostasis' signal to visually show the biological
     spikes riding and "drowning" on top of the massive 2Hz hardware sine wave.
   - Middle Panel (Orthogonal Veto): Plot the KSM metrics for 'homeostasis' vs
     'corrosion'. The corrosion line should trace the homeostasis line closely
     (remaining highly stable), proving Mamba's input-dependent gating used the
     healthy camera to mathematically veto the broken electrode drift. Add a
     vertical dashed line at T=50.
   - Bottom Panel (True Crash): Plot the KSM metric for 'toxic_shock'. It should
     remain perfectly flat during the first half (ignoring the massive pump
     wobble artifact), and violently spike exactly at T=50 when the cell
     actually dies and the systems decouple. Add a vertical dashed line at T=50.
5. Add a `__main__` block to set the device, run `train_orthogonal_veto`, and
   then `evaluate_and_plot`.
