We are upgrading to `OrthogonalModalityEncoder` with Proportional Orthogonal Routing. Please make the following updates:

**1. Update `src/models/encoders/orthonogonal_modality_encoder.py`**
Replace the existing fusion class with this implementation that safely handles edge cases in proportional routing:

```python
import torch
import torch.nn as nn

class OrthogonalModalityEncoder(nn.Module):
    """
    Gated Modality Fusion Layer (Mask-to-Gate Projector).
    Implements Proportional Orthogonal Routing.
    """
    def __init__(self, d_in: int, modality_dims: list[int], d_model: int):
        super().__init__()
        self.W_cart = nn.Linear(d_in, d_model, bias=False)
        
        n_modalities = len(modality_dims)
        if d_model < n_modalities:
            raise ValueError("d_model must be >= the number of modalities.")
            
        self.W_gate = nn.Linear(n_modalities, d_model, bias=True)
        
        d_sensor_total = sum(modality_dims)

        with torch.no_grad():
            # Absolute Stasis Default: sigmoid(-10) = ~4.5e-5.
            # Guarantees memory survives 1000+ step voids when a sensor is missing.
            self.W_gate.bias.fill_(-10.0)
            self.W_gate.weight.fill_(0.0)
            
            # Proportional Orthogonal Routing
            current_idx = 0
            remaining_d_model = d_model
            
            for i, dim in enumerate(modality_dims):
                remaining_modalities = n_modalities - i
                
                if i == n_modalities - 1:
                    # Last modality gets all remaining dimensions to avoid rounding gaps
                    chunk_size = remaining_d_model
                else:
                    proportion = dim / d_sensor_total
                    chunk_size = int(d_model * proportion)
                    
                    # Guarantee at least 1 dimension per modality
                    # but leave enough for the remaining modalities
                    max_allowed = remaining_d_model - (remaining_modalities - 1)
                    chunk_size = max(1, min(chunk_size, max_allowed))
                
                start_idx = current_idx
                end_idx = current_idx + chunk_size
                
                # Map this modality's mask bit to its proportional latent chunk
                self.W_gate.weight[start_idx:end_idx, i] = 20.0
                
                current_idx = end_idx
                remaining_d_model -= chunk_size

        self.W_gate.weight.requires_grad = False
        self.W_gate.bias.requires_grad = False

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor):
        latent_x = self.W_cart(x_raw)
        latent_gate = torch.sigmoid(self.W_gate(mask))
        return latent_x, latent_gate
```

2. Update src/models/simulators/sensor_fusion_predictor.py
Update the SensorFusionPredictor initialization to correctly instantiate the new encoder. Change the __init__ signature to accept modality_dims: list[int] = None.

```
from src.models.encoders.fusion import OrthogonalModalityEncoder

    def __init__(self, ssm_type: str, modality_dims: list[int] = None, d_model: int = 64):
        super().__init__()
        self.ssm_type = ssm_type
        
        # Default to the Waddington dataset dimensions if not provided
        if modality_dims is None:
            modality_dims = [20, 10]
            
        d_sensor_total = sum(modality_dims)
        num_modalities = len(modality_dims)

        if ssm_type == "mask_concat":
            # The input dimension is inflated by the mask size
            self.fusion = OrthogonalModalityEncoder(
                d_in=d_sensor_total + num_modalities, 
                modality_dims=modality_dims, 
                d_model=d_model
            )
        else:
            self.fusion = OrthogonalModalityEncoder(
                d_in=d_sensor_total, 
                modality_dims=modality_dims, 
                d_model=d_model
            )
```

3. write unit test verifying the edge case modality_dims=[1000, 2] with d_model=64 properly allocates exactly 63 channels to Modality 0, and 1 channel to Modality 1 without throwing any errors.

