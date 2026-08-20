import math
import torch
import torch.nn as nn


class BiologicalSpectrogramInit(nn.Module):
    """
    Continuous-Time Spectrogram (Matrix A Initializer)

    Initializes the State Space Model's Matrix A using log-spaced negative real eigenvalues.
    Explicitly maps these initialization frequencies to target biological timescales ranging from
    high-frequency electrophysiology (e.g., 20,000 Hz) to slow morphological drift (e.g., 0.0001 Hz).

    The matrix is parameterized as:
        A = -exp(A_log)
    to guarantee system stability (i.e., strictly negative real eigenvalues: λ_i < 0).

    Mathematical Formulation:
        Let f_min and f_max be the minimum and maximum biological frequencies.
        We generate frequencies f_i log-spaced between f_min and f_max.
        By default, we set the eigenvalues λ_i = -f_i.
        If `use_angular_freq=True`, we use angular frequency: λ_i = -2π f_i.
        Then we store A_log = ln(|λ_i|).
        The forward pass returns A = -exp(A_log) = λ_i.

    Args:
        d_state (int): The state dimension (number of frequencies/eigenvalues to track).
        min_freq (float): Minimum biological frequency in Hz. Default is 0.0001 Hz.
        max_freq (float): Maximum biological frequency in Hz. Default is 20,000 Hz.
        use_angular_freq (bool): Whether to map the frequency in Hz directly (λ = -f)
            or using angular frequency (λ = -2π f). Default is False.
    """

    def __init__(
        self,
        d_state: int,
        min_freq: float = 0.0001,
        max_freq: float = 20000.0,
        use_angular_freq: bool = False,
    ):
        super().__init__()
        self.d_state = d_state
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.use_angular_freq = use_angular_freq

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

        # A_log initialized as a trainable parameter
        self.A_log = nn.Parameter(log_f_i)

    def forward(self) -> torch.Tensor:
        """
        Returns the stable Matrix A diagonal.
        
        Returns:
            torch.Tensor: Shape (d_state,), where A = -exp(A_log)
        """
        return -torch.exp(self.A_log)
