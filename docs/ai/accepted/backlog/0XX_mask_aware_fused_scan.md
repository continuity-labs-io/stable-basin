In src/models/ssm/mask_aware_mamba.py, there is a protective try/except block guarding the Mamba2 import. Because local development on Apple Silicon does not support CUDA, this import quietly fails in the background. When it fails, the code gracefully but fatally falls back to nn.Identity().

There is a second, deeper architectural issue in that same file. If you look at the forward pass of MaskAwareMamba, the gating mechanism simply concatenates the continuous mask to the input features. This means masr_mamba is accidentally executing the exact Mask Concatenation baseline that your paper explicitly critiques! Relying on the off-the-shelf library forces the model back into the representation decay trap because you cannot natively intercept the continuous-time Δt matrix.

Change the architecture so MaskAwareMamba acts as a dynamic bridge: 

1. ingest the dense clinical tensors, 
2. pack them into a sparse event stream on the fly, 
3. route them through your blistering fast MaskAwareFusedScan Triton kernel, and 
4. unpack them back into a dense timeline for the loss function.

We give specific steps below:

Context Files: src/data/async_event_packer.py
Prompt:

```
Please append the following helper function to the end of the file. This function will dynamically pack dense, batched tensors into ragged event streams during the forward pass, allowing our standard dense training loops to seamlessly interact with asynchronous continuous-time kernels.

def dynamic_pack_events(x_raw: torch.Tensor, mask: torch.Tensor, dt_resolution: float = 1.0):
    """
    Packs dense batched tensors into ragged event tensors for Triton execution.
    """
    B, seq_len, Dim = x_raw.shape
    device = x_raw.device

    events_list = []
    lengths = []
    
    for b in range(B):
        time_idx, sensor_idx = torch.nonzero(mask[b] > 0, as_tuple=True)
        vals = x_raw[b, time_idx, sensor_idx]
        timestamps = time_idx.float() * dt_resolution
        sensor_ids = sensor_idx.float()
        
        events_list.append(torch.stack([vals, sensor_ids, timestamps], dim=-1))
        lengths.append(len(vals))

    max_len = max(lengths) if lengths else 0
    padded_events = torch.zeros(B, max_len, 3, device=device)
    event_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

    for b, ev in enumerate(events_list):
        L = lengths[b]
        if L > 0:
            padded_events[b, :L, :] = ev
            event_mask[b, :L] = True

    return padded_events, event_mask
```

Context Files: src/models/ssm/mask_aware_mamba.py
Prompt:

```
Please completely overwrite this file with the following code. We are removing the external mamba_ssm dependency entirely. This module will now serve as a hardware-accelerated dense-to-sparse bridge, natively executing our AsyncMaskAwareSSM Triton kernel to guarantee exact Latent Stasis (A_bar = I) during periods of sensor unobservability.

import torch
import torch.nn as nn
from src.models.ssm.async_ssm import AsyncMaskAwareSSM
from src.models.ssm.state_unpacker import unpack_to_dense
from src.data.async_event_packer import dynamic_pack_events

class MaskAwareMamba(nn.Module):
    """
    Hardware-Accelerated Mask-Aware Engine.
    Dynamically bridges dense clinical tensors with the asynchronous 
    MaskAwareFusedScan Triton kernel to enforce O(N) Latent Stasis.
    """
    def __init__(self, input_dim: int, d_model: int = 256, d_state: int = 16, mask_aware: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.d_state = d_state
        self.mask_aware = mask_aware
        
        # True hardware-accelerated physics engine
        self.ssm = AsyncMaskAwareSSM(dim=input_dim, d_state=d_state)
        
        # Map the flattened thermodynamic state to the required latent space
        self.out_proj = nn.Linear(input_dim * d_state, d_model)

    def forward(self, x, mask=None, return_hidden=False):
        if mask is None:
            mask = torch.ones_like(x)

        B, seq_len, _ = x.shape

        # 1. Pack dense tensors into O(N) sparse event streams
        events, event_mask = dynamic_pack_events(x, mask)

        # 2. Execute blazing fast Triton Fused Scan (Latent Stasis physics)
        h_sparse = self.ssm(events, event_mask)

        # 3. Unpack back to dense time grid via Zero-Order Hold
        h_dense = unpack_to_dense(h_sparse, events, event_mask, seq_len=seq_len, dt_resolution=1.0)

        # 4. Flatten and project to standard d_model embedding
        h_flat = h_dense.view(B, seq_len, self.input_dim * self.d_state)
        h_out = self.out_proj(h_flat)

        if return_hidden:
            return None, None, h_out
        return h_out

    def get_hidden_states(self, x, mask=None):
        _, _, hidden_states = self.forward(x, mask=mask, return_hidden=True)
        return hidden_states
```

Context Files: src/harness/sensor_fusion_predictor.py
Prompt:

In the `__init__` method, please update the initialization logic for the `masr_mamba` condition to utilize the raw physical sensor dimension rather than the projected latent space:
`elif ssm_type == "masr_mamba":`
`    self.ssm = MaskAwareMamba(input_dim=d_sensor_total, d_model=d_model, mask_aware=True)`

In the `forward` method, please update the routing logic for the `masr_mamba` condition. We need to bypass the dense `OrthogonalModalityEncoder` entirely, passing the raw telemetry directly into our new Triton bridge:
`elif self.ssm_type == "masr_mamba":`
`    h = self.ssm.get_hidden_states(x_raw, mask=mask)`


dependencies:
  - triton>=2.1.0
