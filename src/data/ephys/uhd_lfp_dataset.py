import torch
from torch.utils.data import IterableDataset


class ContinuousLFPDataset(IterableDataset):
    """
    A continuous LFP dataset that maps macroscopic electromagnetic fields
    of a 2D UHD-CMOS microelectrode array.
    """

    def __init__(self, time_steps: int = 500, grid_size: int = 64, encoder=None, return_hidden: bool = True, device=None):
        super().__init__()
        self.time_steps = time_steps
        self.grid_size = grid_size
        self.encoder = encoder
        self.return_hidden = return_hidden
        self.device = device if device else torch.device("cpu")

    def __iter__(self):
        while True:
            # Create spatial coordinates
            x = torch.linspace(0, 10, self.grid_size)
            y = torch.linspace(0, 10, self.grid_size)
            Y, X = torch.meshgrid(y, x, indexing="ij")

            # Time vector
            t = torch.linspace(0, 5, self.time_steps)

            # Expand dimensions to [time_steps, grid_size, grid_size]
            X = X.unsqueeze(0).expand(self.time_steps, -1, -1)
            Y = Y.unsqueeze(0).expand(self.time_steps, -1, -1)
            T = t.view(-1, 1, 1).expand(-1, self.grid_size, self.grid_size)

            # Generate continuous 2D traveling wave (V_e)
            k_x, k_y = 1.0, 1.5  # Spatial frequencies
            w = 2.0  # Temporal frequency
            traveling_wave = torch.sin(k_x * X + k_y * Y - w * T)

            # Generate 1/f noise approximation (pink noise)
            white_noise = torch.randn(self.time_steps, self.grid_size, self.grid_size)
            fft_noise = torch.fft.rfft(white_noise, dim=0)
            freqs = torch.fft.rfftfreq(self.time_steps).view(-1, 1, 1)
            freqs = torch.clamp(freqs, min=1e-5)  # Avoid division by zero
            fft_noise = fft_noise / torch.sqrt(freqs)
            pink_noise = torch.fft.irfft(fft_noise, n=self.time_steps, dim=0)

            # Normalize pink noise
            pink_noise = (pink_noise - pink_noise.mean()) / (pink_noise.std() + 1e-8)

            # Combine wave and biological 1/f noise
            noise_amplitude = 0.2
            V_e = traveling_wave + noise_amplitude * pink_noise

            # Add channel dimension: [time_steps, 1, grid_size, grid_size]
            V_e = V_e.unsqueeze(1)

            # Calculate electric field gradients E = -∇V_e
            # Compute spatial derivatives in Y and X directions (dims 2 and 3)
            grad_y, grad_x = torch.gradient(V_e, dim=(2, 3))

            # Negative gradient
            E_y = -grad_y
            E_x = -grad_x

            # Stack gradients to form a 2-channel continuous tensor
            # Shape: [time_steps, 2, grid_size, grid_size]
            E = torch.cat([E_x, E_y], dim=1)

            # Apply spatial encoder on-the-fly to extract geometric priors
            if self.encoder is not None:
                # Expand to batch dimension for encoder: [1, Time, 2, grid, grid]
                E_batched = E.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if self.return_hidden:
                        _, E_encoded = self.encoder(E_batched, return_hidden=True)
                    else:
                        E_encoded = self.encoder(E_batched, return_hidden=False)
                # Remove batch dimension
                E = E_encoded.squeeze(0).cpu()

            # Generate mock visual stimulus embedding (768-D vector)
            visual_stimulus_embedding = torch.randn(768)

            yield E, visual_stimulus_embedding
