Context: I am working on the `stable-basin` repository to build continuous-time measurement metrics for the robust mouse rejuvenation (RMR) project. I need to adapt the spectral analysis logic from the `brain-gen` codebase to measure the loss of biological entrainment and aging-related spectral decoherence.

Task: Extract the spectral evaluation metrics—specifically Power Spectral Density (PSD) calculation, the 1/f power-law decay slope, and cross-spectral density/coherence—from the `brain-gen` evaluation stack (e.g., `brain_gen/eval/rollout_metrics.py`).

Destination: Port this logic into a new module located at `src/metrics/spectral.py`.

Requirements:
1. Isolate the core mathematical logic for calculating the PSD over continuous arrays, extracting the 1/f slope, and computing the cross-spectral density matrix between channels.
2. Provide a rough scaffolding or structural placeholder method for Phase-Amplitude Coupling (PAC) so the hook is there for later.
3. Completely strip out any references to discrete tokenization, vocabulary sizes, codebooks, or language model cross-entropy. The functions must only operate on standard continuous-time numpy arrays (e.g., shape `[batch, channels, time]`).
4. Execution over perfection: This is a rough v1 to secure the architectural scaffolding. Ignore extreme numerical precision, windowing optimizations, filter artifacts, or edge-case handling for now. We just need the pipeline structurally in place.
5. Include standard, calm standard-library logging (e.g., `logger.info("Computing 1/f spectral decay for entrainment metric.")`).

