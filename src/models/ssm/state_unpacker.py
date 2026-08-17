import torch

def unpack_to_dense(
    h_sparse: torch.Tensor, 
    events: torch.Tensor, 
    event_mask: torch.Tensor, 
    seq_len: int, 
    dt_resolution: float = 1.0
) -> torch.Tensor:
    """
    Maps sparse, event-driven hidden states to a uniform dense timeline.
    
    Args:
        h_sparse: [Batch, Max_Events, Dim, D_State]
        events: [Batch, Max_Events, 3] where events[:, :, 2] is the timestamp
        event_mask: [Batch, Max_Events] boolean mask of valid events
        seq_len: Target length of the dense sequence
        dt_resolution: Temporal resolution for the dense grid
        
    Returns:
        h_dense: [Batch, seq_len, Dim, D_State] mapped with Zero-Order Hold
    """
    B, max_events, Dim, D_State = h_sparse.shape
    device = h_sparse.device
    
    # 1. Compute target dense time indices
    time_indices = (events[:, :, 2] / dt_resolution).long()
    time_indices = torch.clamp(time_indices, 0, seq_len - 1)
    
    # 2. Scatter the latest event index into the dense timeline
    valid_mask = event_mask.bool()
    event_indices = torch.arange(max_events, device=device).unsqueeze(0).expand(B, max_events)
    
    # Invalid events scatter -1
    val_to_scatter = torch.where(valid_mask, event_indices, torch.tensor(-1, device=device))
    
    dense_event_idx = torch.full((B, seq_len), -1, dtype=torch.long, device=device)
    
    # Use amax to resolve collisions (multiple events in the same time bin) by keeping the latest one.
    # We use scatter_reduce_ which is available in modern PyTorch versions.
    dense_event_idx.scatter_reduce_(
        1, 
        time_indices, 
        val_to_scatter, 
        reduce="amax",
        include_self=True
    )
    
    # 3. Cumulative max for Zero-Order Hold (forward fill)
    # Since event indices are strictly increasing over time, cummax effectively propagates
    # the last seen event index forward across empty time bins.
    filled_event_idx, _ = dense_event_idx.cummax(dim=1)
    
    # 4. Gather the actual hidden states
    # Prepend a zero-state for time bins before any event has occurred
    h_zero = torch.zeros(B, 1, Dim, D_State, device=device)
    h_padded = torch.cat([h_zero, h_sparse], dim=1) # [B, 1 + max_events, Dim, D_State]
    
    # Shift indices by 1 to map -1 to 0 (the zero state)
    gather_idx = (filled_event_idx + 1).view(B, seq_len, 1, 1).expand(B, seq_len, Dim, D_State)
    h_dense = torch.gather(h_padded, 1, gather_idx)
    
    return h_dense
