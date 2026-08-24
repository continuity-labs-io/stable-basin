import torch
import logging
from src.core.substrate import SubstrateFactory

logger = logging.getLogger(__name__)


def get_optimal_device(verbose: bool = False, allow_mps: bool = True, backend: str = "pytorch"):
    """
    Detects and returns the best available PyTorch device (CUDA, MPS, or CPU),
    or initializes the JAX backend if requested.

    Args:
        verbose (bool): If True, prints the selected device.
        allow_mps (bool): If False, ignores MPS and falls back to CPU.
        backend (str): The backend to use ("pytorch" or "jax").

    Returns:
        torch.device or str: The selected PyTorch device or "jax".
    """
    substrate = SubstrateFactory.get_substrate(allow_mps=allow_mps, backend=backend)
    device = substrate.device

    if backend == "jax":
        if verbose:
            logger.info("JAX backend initialized.")
        return "jax"

    if verbose:
        logger.info(f"Using device: {device}")

    return device
