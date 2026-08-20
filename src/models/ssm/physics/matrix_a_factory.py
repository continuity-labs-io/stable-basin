import math
import torch
import torch.nn as nn

class LogSpacedAInit(nn.Module):
    """
    Continuous-Time Spectrogram (Matrix A Initializer)

    Initializes the State Space Model's Matrix A using log-spaced negative real eigenvalues.
    Explicitly maps these initialization frequencies to target biological timescales ranging from
    high-frequency electrophysiology (e.g., 20,000 Hz) to slow morphological drift (e.g., 0.0001 Hz).

    The matrix is parameterized as:
        A = -exp(A_log)
    to guarantee system stability (i.e., strictly negative real eigenvalues: λ_i < 0).

    Args:
        shape (tuple): The required shape of the parameter, e.g. (d_state,) or (d_model, d_state).
            The log-spacing is applied across the last dimension (d_state).
        min_freq (float): Minimum biological frequency in Hz. Default is 0.0001 Hz.
        max_freq (float): Maximum biological frequency in Hz. Default is 20,000 Hz.
        use_angular_freq (bool): Whether to map the frequency in Hz directly (λ = -f)
            or using angular frequency (λ = -2π f). Default is False.
    """
    def __init__(
        self,
        shape: tuple[int, ...],
        min_freq: float = 0.0001,
        max_freq: float = 20000.0,
        use_angular_freq: bool = False,
    ):
        super().__init__()
        self.shape = shape
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.use_angular_freq = use_angular_freq

        if not shape:
            raise ValueError("Shape cannot be empty.")

        d_state = shape[-1]

        if min_freq <= 0 or max_freq <= 0:
            raise ValueError("Frequencies must be strictly positive.")
        if min_freq >= max_freq:
            raise ValueError("min_freq must be less than max_freq.")

        # Log-space frequencies f_i from f_min to f_max
        log_f_min = math.log(min_freq)
        log_f_max = math.log(max_freq)
        
        # log_f_i has shape (d_state,)
        log_f_i = torch.linspace(log_f_min, log_f_max, d_state)

        if use_angular_freq:
            # ln(2 * pi * f_i) = ln(2 * pi) + ln(f_i)
            log_f_i = log_f_i + math.log(2 * math.pi)

        # Broadcast to full shape (e.g., if shape is (d_model, d_state))
        expanded_log_f_i = log_f_i.expand(shape).contiguous()

        # A_log initialized as a trainable parameter
        self.A_log = nn.Parameter(expanded_log_f_i)

    def forward(self) -> torch.Tensor:
        """
        Returns the stable Matrix A.
        """
        return -torch.exp(self.A_log)

class RandomAInit(nn.Module):
    """
    Random Continuous-Time Matrix A Initializer
    Initializes A_log uniformly to achieve diverse timescales.
    """
    def __init__(self, shape: tuple[int, ...], a_scale: float = 0.5, a_shift: float = 0.1):
        super().__init__()
        self.shape = shape
        
        # A_log initialized uniformly
        self.A_log = nn.Parameter(torch.log(torch.rand(shape) * a_scale + a_shift))

    def forward(self) -> torch.Tensor:
        return -torch.exp(self.A_log)

def create_a_matrix(init_type: str, shape: tuple[int, ...], **kwargs) -> nn.Module:
    """
    Factory function for State Space Model Matrix A initialization.
    
    Args:
        init_type (str): Either "random" or "log_spaced".
        shape (tuple): Shape of the A matrix (typically (d_state,) or (d_model, d_state)).
        
    Returns:
        nn.Module: An initializer module with a `.A_log` parameter and a `.forward()` returning `A`.
    """
    if init_type == "log_spaced":
        return LogSpacedAInit(shape, **kwargs)
    elif init_type == "random":
        return RandomAInit(shape, **kwargs)
    else:
        raise ValueError(f"Unknown A matrix init_type: {init_type}. Must be 'random' or 'log_spaced'.")
