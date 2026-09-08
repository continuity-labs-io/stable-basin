import torch
import scipy.signal
import numpy as np
import logging

logger = logging.getLogger("DiagnosticLogger")

class SpectralMetrics:
    def calculate_psd(self, tensor_seq: torch.Tensor, sampling_rate: float):
        """
        Calculates the Power Spectral Density (PSD) using a standard real FFT.
        """
        # Fallback for missing sensor data
        tensor_seq = torch.nan_to_num(tensor_seq, nan=0.0)

        time_dim = 0
        n = tensor_seq.shape[time_dim]

        # Handle flatlines gracefully
        if torch.var(tensor_seq, dim=time_dim).mean() < 1e-8:
            freqs = torch.fft.rfftfreq(n, d=1.0 / sampling_rate)
            fft_shape = list(tensor_seq.shape)
            fft_shape[time_dim] = n // 2 + 1
            power = torch.zeros(fft_shape, device=tensor_seq.device)
            logger.debug("Flatline detected in PSD. Returning zero power array.")
            return freqs, power

        # Detrend / Mean-center to remove DC offset
        tensor_seq = tensor_seq - tensor_seq.mean(dim=time_dim, keepdim=True)

        fft_out = torch.fft.rfft(tensor_seq, dim=time_dim)
        power = torch.abs(fft_out) ** 2 / n
        freqs = torch.fft.rfftfreq(n, d=1.0 / sampling_rate)

        logger.debug("Calculated PSD array.")
        return freqs, power

    def calculate_plv(self, seq_a: torch.Tensor, seq_b: torch.Tensor):
        """
        Calculates the Phase-Locking Value (PLV) between two sequences.
        """
        seq_a = torch.nan_to_num(seq_a, nan=0.0)
        seq_b = torch.nan_to_num(seq_b, nan=0.0)

        time_dim = 0
        min_steps = min(seq_a.shape[time_dim], seq_b.shape[time_dim])
        
        if seq_a.dim() == 1:
            seq_a = seq_a[:min_steps]
            seq_b = seq_b[:min_steps]
        else:
            seq_a = seq_a[:min_steps, ...]
            seq_b = seq_b[:min_steps, ...]

        device = seq_a.device
        np_a = seq_a.detach().cpu().numpy()
        np_b = seq_b.detach().cpu().numpy()

        analytic_a = scipy.signal.hilbert(np_a, axis=time_dim)
        analytic_b = scipy.signal.hilbert(np_b, axis=time_dim)

        # Add a microscopic epsilon to prevent angle singularities on zero
        # magnitude
        analytic_a += 1e-8
        analytic_b += 1e-8

        phase_a = torch.tensor(np.angle(analytic_a), device=device)
        phase_b = torch.tensor(np.angle(analytic_b), device=device)

        phase_diff = phase_a - phase_b
        complex_phase_diff = torch.exp(1j * phase_diff)
        plv = torch.abs(torch.mean(complex_phase_diff, dim=time_dim))

        logger.debug("Calculated PLV array.")
        return plv

    def calculate_cfc_pac(self, slow_seq: torch.Tensor, fast_seq: torch.Tensor):
        """
        Cross-Frequency Coupling (CFC) / Phase-Amplitude Coupling (PAC) via Mean
        Vector Length (MVL). Measures hierarchical enslavement: the degree to
        which a slow macroscopic order parameter restricts the
        amplitude/variance of fast microscopic variables.
        """
        # NaN contagion fallback
        slow_seq = torch.nan_to_num(slow_seq, nan=0.0)
        fast_seq = torch.nan_to_num(fast_seq, nan=0.0)

        time_dim = 0
        min_steps = min(slow_seq.shape[time_dim], fast_seq.shape[time_dim])
        
        # Explicitly enforce identical temporal dimensions to prevent mismatched
        # sequence errors
        if slow_seq.dim() == 1:
            slow_seq = slow_seq[:min_steps]
            fast_seq = fast_seq[:min_steps]
        else:
            slow_seq = slow_seq[:min_steps, ...]
            fast_seq = fast_seq[:min_steps, ...]

        # Assert identical temporal dimensions after trimming
        assert slow_seq.shape[time_dim] == fast_seq.shape[time_dim], "Temporal dimensions must be identical for PAC computation."

        device = slow_seq.device
        np_slow = slow_seq.detach().cpu().numpy()
        np_fast = fast_seq.detach().cpu().numpy()

        # Extract the analytic signal for both sequences using the Hilbert
        # transform
        analytic_slow = scipy.signal.hilbert(np_slow, axis=time_dim)
        analytic_fast = scipy.signal.hilbert(np_fast, axis=time_dim)

        # The instantaneous phase of the slow macroscopic variable (the top-down
        # prior)
        theta_slow = torch.tensor(np.angle(analytic_slow), device=device)
        
        # The instantaneous amplitude envelope of the fast microscopic variable
        # (the enslaved state)
        A_fast = torch.tensor(np.abs(analytic_fast), device=device)

        # Compute the complex composite signal: z(t) = A_fast(t) * exp(i *
        # theta_slow(t))
        z_t = A_fast * torch.exp(1j * theta_slow)

        # The thermodynamic coupling metric is the absolute mean vector length
        pac_mvl = torch.abs(torch.mean(z_t, dim=time_dim))

        logger.debug("Calculated CFC-PAC array via MVL.")
        return pac_mvl
