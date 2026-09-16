Create a new script `src/demo/hssm_toy_model.py`. First, we need to generate the
synthetic 'Drowning Signal' multiscale data.

Write a PyTorch class `ToyBiologicalEnvironment`. It should have a method
`generate_batch(batch_size, scenario="homeostasis", device="cpu")`:

1. Time steps: 100 frames for Optics (100Hz), 20,000 frames for GEVI (20kHz).
2. The Pump Artifact: Generate a massive 2Hz sine wave. Add its 100Hz version
   (amplitude 2.0) across all 768 dimensions of the Optical tensor
   `[batch_size, 100, 768]`. Add its 20kHz version (amplitude 50.0) to the GEVI
   tensor `[batch_size, 1, 20000]`.
3. The Biology: Inject sparse 1ms spikes (amplitude 100.0) into the GEVI tensor
   (about 1% probability per 200-step window).
4. Scenario 'homeostasis': Returns the normal data with pump artifact and
   biological spikes.
5. Scenario 'corrosion' (Hardware Failure): At T=50 frames (frame 10000 in
   GEVI), add a massive random-walk baseline drift to the GEVI tensor. Optical
   remains perfectly normal.
6. Scenario 'toxic_shock' (Biological Crash): At T=50 frames, the GEVI
   biological spikes stop completely, and the Optical tensor experiences a
   variance explosion (add normal noise with std=5.0).

Return `optical_tensor` and `gevi_tensor` as float32.

Requirements:

- Include basic support for apple MPS if possible.
- Ensure adequate documentation.
- Avoid magic numbers by using constants with meaningful names.
