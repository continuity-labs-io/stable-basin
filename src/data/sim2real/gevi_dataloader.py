import torch
from src.config import settings


class GEVIDataloader:
    """
    Simulates a high-frequency bioelectric data stream (GEVI).
    Generates synthetic membrane potentials, action potentials, 
    and optionally injects a variance explosion anomaly.

    Args:
        gevi_sample_rate (int): The sampling rate of the high-frequency GEVI data in Hz.
        target_clock_hz (int): The target framerate to pool the data down to in Hz.
        baseline_mv (float): The baseline membrane potential in millivolts (mV).
        noise_std (float): The standard deviation of the thermal noise in millivolts (mV).
        spike_prob (float): The probability of an action potential spike occurring
            at any given step.
        spike_mv (float): The amplitude of an action potential spike in millivolts (mV).
        anomaly_start_frame (int): The optical frame index at which to start injecting
            the variance explosion anomaly.
        anomaly_noise_std (float): The standard deviation of the variance explosion noise
            in millivolts (mV).
    """

    def __init__(
        self,
        gevi_sample_rate=settings.GEVI_HZ,
        target_clock_hz=settings.OPTICS_HZ,
        baseline_mv=-70.0,
        noise_std=2.0,
        spike_prob=0.01,
        spike_mv=100.0,
        anomaly_start_frame=6,
        anomaly_noise_std=40.0,
    ):
        self.compression_ratio = int(gevi_sample_rate / target_clock_hz)
        self.baseline_mv = baseline_mv
        self.noise_std = noise_std
        self.spike_prob = spike_prob
        self.spike_mv = spike_mv
        self.anomaly_start_frame = anomaly_start_frame
        self.anomaly_noise_std = anomaly_noise_std

    def generate_synthetic_gevi(self, batch_size, target_time_steps, device, is_healthy=True):
        total_steps = target_time_steps * self.compression_ratio
        tensor = torch.full((batch_size, 1, total_steps), self.baseline_mv, device=device)
        tensor += torch.randn_like(tensor) * self.noise_std

        # sparse action potential spikes
        mask = torch.rand_like(tensor) < self.spike_prob
        tensor[mask] += self.spike_mv

        if not is_healthy:
            start_idx = self.anomaly_start_frame * self.compression_ratio
            if start_idx < total_steps:
                variance_injection = (
                    torch.randn((batch_size, 1, total_steps - start_idx), device=device)
                    * self.anomaly_noise_std
                )
                tensor[:, :, start_idx:] += variance_injection

        return tensor
