import jax
import logging

logger = logging.getLogger(__name__)


def torch_to_jax(tensor):
    """
    Converts a PyTorch tensor to a JAX array using DLPack for zero-copy memory transfers.

    Args:
        tensor: A PyTorch tensor to be converted.

    Returns:
        A JAX array referencing the same underlying memory.
    """
    # Import torch locally to prevent breaking headless TPU/GPU execution
    import torch

    logger.debug("Executing zero-copy DLPack transfer.")

    # Check if input is a PyTorch tensor, to provide a clear error message
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    return jax.dlpack.from_dlpack(tensor)
