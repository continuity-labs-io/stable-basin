Implement SyntheticBiohybridDataset for compatibility with Science Corp NYX 512 / LUX 2K paradigm.

Context Files to Load:

src/config.py

src/data/ephys/uhd_lfp_dataset.py

src/data/synthetic/gevi_dataloader.py

src/data/async_event_packer.py

Prompt Text:

Plaintext
Create a new dataset module at src/data/ephys/synthetic_biohybrid.py that generates synthetic wideband electrophysiology for biohybrid neural interfaces (such as Science Corp's NYX 512 / LUX 2K paradigm).

Requirements:
1. Architecture & Class Design:
   - Name the class `SyntheticBiohybridDataset`, inheriting from `torch.utils.data.Dataset`.
   - The dataset must generate multi-channel electrophysiology across `num_channels` (default: 64, configurable up to 512).
   - Sampling rate default: 20000 Hz (`sample_rate_hz=20000`).
   - Sequence length default: 2048 time steps per sample chunk.

2. Generative Dynamics:
   - Low-Frequency LFP Layer (< 300 Hz):
     * Generate 2 to 4 continuous macroscopic rhythms (theta: 4-8 Hz, beta: 15-30 Hz, gamma: 30-80 Hz).
     * Project these through a spatial mixing matrix with distance-dependent exponential decay across channels to enforce spatial covariance.
     * Add 1/f pink noise generated in the frequency domain.
   - High-Frequency Action Potentials (Spikes, 300 Hz - 5 kHz):
     * Model stochastic spike generation using an inhomogeneous Poisson process where instantaneous firing rate is coupled to the phase of the dominant LFP oscillation (spike-field coherence).
     * Synthesize realistic bi-phasic/tri-phasic extracellular action potential waveforms (duration: ~1.0 to 1.5 ms) and convolve them onto the active channels.
   - Optogenetic Write Modulation (Watcher Opsin Actuation):
     * Accept an optional binary optical stimulation vector of shape [time_steps, num_channels] or simulate sparse optical pulses (pulse width: ~5 ms).
     * When an optical pulse occurs on a channel, inject an immediate high-probability evoked action potential with minimal latency jitter (~1 ms) and perturb the local LFP phase.

3. Output Tensor Structure:
   - The `__getitem__` method must return a dictionary:
     * `wideband`: torch.Tensor of shape [seq_len, num_channels] (combined raw trace in microvolts, float32).
     * `lfp`: torch.Tensor of shape [seq_len, num_channels] (low-pass filtered / ground-truth field potential).
     * `spikes`: torch.Tensor of shape [seq_len, num_channels] (binary ground-truth spike raster).
     * `optogenetic_actuation`: torch.Tensor of shape [seq_len, num_channels] (optical stimulus intensity [0, 1]).
     * `mask`: torch.Tensor of shape [seq_len, num_channels] (default all ones, supporting simulated channel dropouts).

4. Coding Standards:
   - Use standard logging (logger = logging.getLogger(__name__)) with subdued, professional messages. No dramatic logging or emojis.
   - Ensure all operations are vectorized using PyTorch where possible for fast execution.
   - Type annotate all signatures using jaxtyping / beartype where appropriate to maintain consistency with the rest of Stable Basin.
