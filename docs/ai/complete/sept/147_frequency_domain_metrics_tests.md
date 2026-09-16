**Role Context:** Principal Software Engineer / AI Architect

Ensure the new frequency domain metrics have some tests.

## I. Mathematical Corner Cases (The Tripwires)

### 1. Power Spectral Density (PSD)
- **The Flatline (Zero Variance):** If a sensor drops or a cell dies, the input sequence becomes a constant (e.g., all 0.0 or all 5.0).
  - **Failure Mode:** `torch.fft` can amplify floating-point noise, or normalization steps might divide by zero.
  - **Expected Behavior:** Should gracefully return a power array of zeros without throwing NaNs.
- **The DC Offset:** Biological signals often drift heavily off zero (e.g., baseline membrane voltage).
  - **Failure Mode:** The DC component (0 Hz) dominates the power spectrum, squashing all biological resonant frequencies into invisibility.
  - **Expected Behavior:** The function should explicitly detrend or mean-center the tensor before applying the FFT.

### 2. Phase-Locking Value (PLV)
- **The Undefined Phase (Zero Amplitude):** The Hilbert transform extracts instantaneous phase, but what is the phase of a flat line?
  - **Failure Mode:** Calculating the angle of a zero-magnitude complex number results in NaN, which will poison the entire batch average.
  - **Expected Behavior:** The function must detect near-zero amplitudes and mask them, or add a microscopic $\epsilon$ (e.g., 1e-8) to prevent division/angle singularities.
- **Perfect Anti-Correlation:** `seq_a` is exactly 180° out of phase with `seq_b` (e.g., $sin(x)$ vs $-sin(x)$).
  - **Expected Behavior:** PLV measures consistency of the phase difference, not the difference itself. A consistent 180° difference should yield a PLV of exactly 1.0, not 0.0.

### 3. Cross-Frequency Coupling (CFC / PAC)
- **Non-Sinusoidal Slow Waves:** Real biological macro-states are rarely perfect sine waves; they are often asymmetric saw-tooths.
  - **Failure Mode:** Sharp edges in the slow wave introduce high-frequency harmonics during the Hilbert transform, creating "spurious" coupling that looks like PAC but is actually just a math artifact.
  - **Expected Behavior:** The testing suite should pass an asymmetric wave and verify the MVL algorithm doesn't falsely report massive coupling.
- **Mismatched Sequence Lengths:** Due to hardware packet drops or multi-rate polling (e.g., optical vs. electrophysiology).
  - **Expected Behavior:** The function must explicitly assert that `slow_seq` and `fast_seq` have identical temporal dimensions before executing the tensor multiplication.

## II. Substrate and Data Structure Corner Cases

- **The NaN Contagion:** If 15% of the sensors drop out (as tested in the Blind Reach route), the input tensor will contain NaNs. FFTs and Hilbert transforms will instantly propagate a single NaN across the entire temporal sequence.
  - **Expected Behavior:** The suite must include a `torch.nan_to_num()` fallback or explicitly drop masked channels before frequency-domain processing.
- **Even vs. Odd Sequence Lengths:** The mathematical definition of the discrete Hilbert transform via FFT handles the Nyquist frequency differently depending on whether the total number of time steps $N$ is even or odd.
  - **Expected Behavior:** Feed the functions a sequence of length 99 and a sequence of length 100. Both must return identical dimensionality without index out-of-bounds errors.
- **Device Agnosticism:** Operations like `torch.fft` and `torch.angle` occasionally have different edge-case behaviors on CPU versus MPS (Apple Silicon) versus CUDA.
  - **Expected Behavior:** Tensors generated on one device must remain on that device.

## III. The Regression Testing Protocol

To push this into our CI/CD pipeline (make preflight), we should architect a dedicated `test_spectral_telemetry.py` file with the following deterministic regression tests:

### The Identity Test (Ground Truth Calibration):
- Generate a pure 2Hz sine wave and a pure 10Hz sine wave.
- Assert that PSD accurately identifies the 2Hz and 10Hz bins.
- Assert that PLV between two identical 2Hz waves is exactly 1.0.
- Assert that PLV between the 2Hz and 10Hz wave is near 0.0.

### The Enslavement Test (CFC Validation):
- Generate a synthetic signal where a 50Hz high-frequency burst only occurs during the peak of a 2Hz low-frequency wave.
- Assert that `calculate_cfc_pac` returns a strongly positive value ($> 0.8$).
- Generate a control signal where the 50Hz bursts are uniformly distributed. Assert CFC is near 0.0.

### The VRAM Leak Test (Hardware Scaling):
- Feed a massive tensor (e.g., Batch=8, Channels=1024, Time=5000) into the PLV and CFC functions.
- Monitor `torch.mps.current_allocated_memory()` (or CUDA equivalent).
- Assert that the memory footprint remains stable and $O(1)$ without accumulating graph history, verifying the engine can run indefinitely on edge hardware.
