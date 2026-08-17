import torch
from torch.utils.data import Dataset

class AsyncEventPackerDataset(Dataset):
    """
    Wraps a dense, zero-padded dataset and dynamically 'packs' it into a sparse, 
    1-dimensional Event Tensor. Eliminates VRAM waste.
    """
    def __init__(self, base_dataset, dt_resolution=1.0):
        self.base_dataset = base_dataset
        self.dt_resolution = dt_resolution

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        batch = self.base_dataset[idx]
        x_raw = batch["x_raw"]  # [Seq_Len, Dim]
        mask = batch["mask"]    # [Seq_Len, Dim]
        y_true = batch.get("y_true", None)

        # Extract active events (torch.nonzero naturally sorts by time_idx then sensor_idx)
        time_indices, sensor_indices = torch.nonzero(mask > 0, as_tuple=True)
        
        # 1. Extract values
        values = x_raw[time_indices, sensor_indices]
        
        # 2. Compute physical timestamps
        timestamps = time_indices.float() * self.dt_resolution
        
        # 3. Extract Sensor IDs
        sensor_ids = sensor_indices.float()
        
        # Pack into [Num_Events, 3] -> [Value, Sensor_ID, Timestamp]
        events = torch.stack([values, sensor_ids, timestamps], dim=-1)
        
        out = {"events": events, "num_events": len(events)}
        if y_true is not None:
            out["y_true"] = y_true
            
        return out

def ragged_collate_fn(batch):
    """
    Collates ragged event tensors by padding to max_events in the current batch.
    Outputs the critical boolean 'event_mask' used by the downstream models.
    """
    events_list = [item["events"] for item in batch]
    lengths = torch.tensor([item["num_events"] for item in batch])
    max_len = lengths.max().item() if len(lengths) > 0 else 0
    batch_size = len(batch)
    
    padded_events = torch.zeros(batch_size, max_len, 3)
    event_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, ev in enumerate(events_list):
        L = lengths[i].item()
        if L > 0:
            padded_events[i, :L, :] = ev
            event_mask[i, :L] = True
            
    out = {
        "events": padded_events, 
        "event_mask": event_mask, 
        "lengths": lengths
    }
    
    if batch[0].get("y_true") is not None:
        out["y_true"] = torch.stack([item["y_true"] for item in batch])
        
    return out
