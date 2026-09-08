# 🛠️ ENGINEERING TICKET: Implement Foundational Frequency and Phase

**Role Context:** Principal Software Engineer / AI Architect

Feed the following file paths into your context:

src/metrics/metrics.py

src/config.py

Raw Prompt to Execute:

Add two new methods to the ThermodynamicMetrics class in src/metrics/metrics.py: calculate_psd and calculate_plv.

calculate_psd(self, tensor_seq, sampling_rate): Implement Power Spectral Density using Welch's method or a standard real FFT (torch.fft.rfft). Return the frequency bins and the power array.

calculate_plv(self, seq_a, seq_b): Implement the Phase-Locking Value. Compute the analytic signal via the Hilbert transform to extract the instantaneous phase (θ₁ and θ₂) for both sequences. Calculate PLV as the absolute value of the mean of exp(i * (θ₁ - θ₂)).

Implementation constraints: Ensure the mathematical operations are fully vectorized using PyTorch to maintain O(1) VRAM efficiency. Keep all generated Python code calm, precise, and practical. Dial down the intensity of any logging or print statements, utilizing subdued, professional language (e.g., logger.debug('Calculated PLV array.')). Use standard, readable variable names.

Dependencies
- consider scipy (Specifically for scipy.signal.hilbert to safely and efficiently extract the analytic signal for phase calculations).
