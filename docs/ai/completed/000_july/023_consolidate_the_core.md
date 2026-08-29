Let's knock out Suggestion 1: Consolidate the Core. We are injecting the
hardware-agnostic routing logic into Stable Basin Benchmark so it scales
flawlessly on Apple Silicon or Edge GPUs, and we are centralizing the scattered
physics constants.

Here is your pristine, unified src/core/substrate.py to drop into the
repository. It abstracts all the if torch.backends.mps.is_available()
boilerplate.

```
"""
MELD Core Architecture: Hardware Substrate Abstraction
Provides a unified interface for memory management, synchronization, and
hardware-specific mathematical operations across Apple Silicon, NVIDIA CUDA, and CPUs.
"""

import abc
import logging
import torch

logger = logging.getLogger("meld-substrate")

class HardwareSubstrate(abc.ABC):
    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @abc.abstractmethod
    def synchronize(self) -> None:
        pass

    @abc.abstractmethod
    def empty_cache(self) -> None:
        pass

    @abc.abstractmethod
    def current_memory_mb(self) -> float:
        pass


class MPSSubstrate(HardwareSubstrate):
    def __init__(self):
        self._device = torch.device("mps")

    @property
    def device(self) -> torch.device:
        return self._device

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def empty_cache(self) -> None:
        torch.mps.empty_cache()

    def current_memory_mb(self) -> float:
        return torch.mps.current_allocated_memory() / (1024 * 1024)


class CUDASubstrate(HardwareSubstrate):
    def __init__(self):
        self._device = torch.device("cuda")

    @property
    def device(self) -> torch.device:
        return self._device

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def current_memory_mb(self) -> float:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)


class CPUSubstrate(HardwareSubstrate):
    def __init__(self):
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def current_memory_mb(self) -> float:
        return 0.0


class SubstrateFactory:
    _instance = None

    @classmethod
    def get_substrate(cls, allow_mps: bool = True) -> HardwareSubstrate:
        if cls._instance is not None:
            # If the user specifically disabled MPS for a Mamba-2 backward pass, we must honor it
            if not allow_mps and cls._instance.is_mps:
                cls._instance = CPUSubstrate()
            return cls._instance

        if torch.cuda.is_available():
            cls._instance = CUDASubstrate()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and allow_mps:
            cls._instance = MPSSubstrate()
        else:
            cls._instance = CPUSubstrate()

        return cls._instance

def get_optimal_device(verbose: bool = False, allow_mps: bool = True) -> torch.device:
    device = SubstrateFactory.get_substrate(allow_mps).device
    if verbose:
        logger.info(f"[*] Provisioned Hardware Substrate: {str(device).upper()}")
    return device
```
