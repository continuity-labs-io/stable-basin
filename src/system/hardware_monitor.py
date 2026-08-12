import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class HardwareMonitor:
    """
    Monitors peak VRAM allocation to benchmark linear (Mamba) vs quadratic (Transformer)
    scaling efficiencies across varying sequence lengths.

    Note: This is configured to support macOS (using a mock CPU fallback).
    For full hardware acceleration on non-Mac systems, a different Mamba library
    (such as the official `mamba_ssm` with CUDA support) is recommended.

    Args:
        device (torch.device): The PyTorch device to run benchmarks on (e.g., 'cpu', 'cuda').
        num_heads (int): Number of attention heads for the legacy Transformer baseline.
        d_state (int): The state dimension for the Mamba block.
        d_conv (int): The convolution kernel size for the Mamba block.
        expand (int): The expansion factor for the Mamba block.
        base_vram_mb (float): Base memory overhead in MB used when mocking VRAM on CPU.
        bytes_per_float (int): Bytes per float used for mock VRAM calculations (e.g. 4 for FP32).
    """

    def __init__(
        self,
        device,
        num_heads=8,
        d_state=16,
        d_conv=4,
        expand=2,
        base_vram_mb=15.0,
        bytes_per_float=4,
    ):
        self.device = device
        self.num_heads = num_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.base_vram_mb = base_vram_mb
        self.bytes_per_float = bytes_per_float

    def run_scaling_benchmark(self, d_model=832, seq_lengths=None):
        if seq_lengths is None:
            seq_lengths = [100, 500, 1000, 2000, 4000, 8000]

        if self.device.type not in ["cuda", "mps"]:
            logger.info(
                "Warning: Hardware acceleration not available. "
                "Returning mock lists for CPU demonstration."
            )
            mamba_vram = [
                self.base_vram_mb + (L * d_model * self.bytes_per_float * self.d_state / (1024**2))
                for L in seq_lengths
            ]
            transformer_vram = [
                self.base_vram_mb
                + (L * d_model * self.bytes_per_float / (1024**2))
                + (L**2 * self.num_heads * self.bytes_per_float / (1024**2))
                for L in seq_lengths
            ]
            return seq_lengths, mamba_vram, transformer_vram

        attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=self.num_heads, batch_first=True
        ).to(self.device)
        if Mamba is not None:
            mamba = Mamba(
                d_model=d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            ).to(self.device)
        else:
            logger.warning("Warning: mamba_ssm not installed, using mock memory scaling for Mamba.")
            mamba = None

        mamba_vram = []
        transformer_vram = []

        for L in seq_lengths:
            x = torch.randn(1, L, d_model).to(self.device)

            if mamba is not None:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = mamba(x)
                if self.device.type == "cuda":
                    mamba_vram.append(torch.cuda.max_memory_allocated() / (1024**2))
                elif self.device.type == "mps":
                    mamba_vram.append(torch.mps.current_allocated_memory() / (1024**2))
            else:
                mamba_vram.append(
                    self.base_vram_mb
                    + (L * d_model * self.bytes_per_float * self.d_state / (1024**2))
                )

            try:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = attn(x, x, x)
                if self.device.type == "cuda":
                    transformer_vram.append(torch.cuda.max_memory_allocated() / (1024**2))
                elif self.device.type == "mps":
                    transformer_vram.append(torch.mps.current_allocated_memory() / (1024**2))
            except Exception:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
                transformer_vram.append(None)

        return seq_lengths, mamba_vram, transformer_vram
