import torch
import logging

logger = logging.getLogger("DiagnosticLogger")

def _hilbert_transform(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Computes the analytic signal using the Hilbert transform in native PyTorch.
    This replicates scipy.signal.hilbert exactly and prevents CPU syncs.
    """
    N = x.shape[dim]
    Xf = torch.fft.fft(x, dim=dim)
    
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
        
    shape = [1] * x.dim()
    shape[dim] = N
    h = h.view(shape)
    
    return torch.fft.ifft(Xf * h, dim=dim)

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

        analytic_a = _hilbert_transform(seq_a, dim=time_dim)
        analytic_b = _hilbert_transform(seq_b, dim=time_dim)

        # Add a microscopic epsilon to prevent angle singularities on zero
        # magnitude
        analytic_a += 1e-8
        analytic_b += 1e-8

        phase_a = torch.angle(analytic_a)
        phase_b = torch.angle(analytic_b)

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

        # Extract the analytic signal for both sequences using the Hilbert
        # transform natively in PyTorch
        analytic_slow = _hilbert_transform(slow_seq, dim=time_dim)
        analytic_fast = _hilbert_transform(fast_seq, dim=time_dim)

        # The instantaneous phase of the slow macroscopic variable (the top-down
        # prior)
        theta_slow = torch.angle(analytic_slow)
        
        # The instantaneous amplitude envelope of the fast microscopic variable
        # (the enslaved state)
        A_fast = torch.abs(analytic_fast)

        # Compute the complex composite signal: z(t) = A_fast(t) * exp(i *
        # theta_slow(t))
        z_t = A_fast * torch.exp(1j * theta_slow)

        # The thermodynamic coupling metric is the absolute mean vector length
        pac_mvl = torch.abs(torch.mean(z_t, dim=time_dim))

        logger.debug("Calculated CFC-PAC array via MVL.")
        return pac_mvl
