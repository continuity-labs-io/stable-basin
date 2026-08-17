import torch
import warnings
import logging
from torch.utils.data import Dataset, DataLoader
import math

from src.data.waddington_dataset import SyntheticWaddingtonDataset
from src.data.async_event_packer import AsyncEventPackerDataset, ragged_collate_fn
from src.models.ssm.async_ssm import AsyncMaskAwareSSM
from src.models.ssm.state_unpacker import unpack_to_dense
from src.metrics.metrics import ThermodynamicMetrics

class CrashedWaddingtonDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        x_raw = item["x_raw"]
        seq_len = x_raw.shape[0]
        
        # Inject biological crash at frame 150
        for i in range(150, seq_len):
            x_raw[i] = x_raw[i] * math.exp((i - 150) * 1.5)
            
        item["x_raw"] = x_raw
        return item

def test_async_ksm():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    logging.getLogger("DiagnosticLogger").setLevel(logging.CRITICAL)
    torch.manual_seed(42)
    print("--- Running Async KSM Validation ---")
    
    # 1. Base dataset
    base_dataset = SyntheticWaddingtonDataset(size=1, seq_len=300, density=0.1)
    
    # 2. Inject crash
    crashed_dataset = CrashedWaddingtonDataset(base_dataset)
    
    # 3. Pack to async events
    packer_dataset = AsyncEventPackerDataset(crashed_dataset, dt_resolution=1.0)
    loader = DataLoader(packer_dataset, batch_size=1, collate_fn=ragged_collate_fn)
    
    batch = next(iter(loader))
    events = batch["events"]
    event_mask = batch["event_mask"]
    
    print(f"Packed sparse events shape: {events.shape}")
    
    # 4. Asynchronous SSM
    # Dim 30 because Modality 0 is 20D and Modality 1 is 10D
    model = AsyncMaskAwareSSM(dim=30, d_state=16)
    
    # Boost signal for the un-trained model so DMD doesn't flatline (temporal_std < 1e-3)
    with torch.no_grad():
        model.B_proj.normal_(mean=1.0, std=0.2)
        model.A_log.fill_(math.log(0.9))
        
    with torch.no_grad():
        h_sparse = model(events, event_mask)
        print(f"Sparse hidden states shape: {h_sparse.shape}")
        
        # 5. Unpack to dense timeline
        h_dense = unpack_to_dense(h_sparse, events, event_mask, seq_len=300, dt_resolution=1.0)
        print(f"Dense hidden states shape: {h_dense.shape}")
        
        # 6. Flatten for ThermodynamicMetrics
        # h_dense is [Batch, Seq_Len, Dim, D_State] -> [1, 300, 30, 16]
        h_flat = h_dense.view(300, 30 * 16)
        
        metrics = ThermodynamicMetrics(alpha=500.0)
        ksm_scores = metrics.calculate_ksm(h_flat, window_size=10)
        
        # 7. Print and Assert
        pre_crash_avg = sum(ksm_scores[50:140]) / len(ksm_scores[50:140])
        post_crash_avg = sum(ksm_scores[250:300]) / len(ksm_scores[250:300])
        
        print(f"Average KSM pre-crash (frames 50-140): {pre_crash_avg:.4f}")
        print(f"Average KSM post-crash (frames 250-300): {post_crash_avg:.4f}")
        
        assert pre_crash_avg > 0.8, "Pre-crash KSM should be near 1.0 (stable)"
        assert post_crash_avg < 0.2, "Post-crash KSM should drop significantly (unstable)"
        
        print("Validation Successful: Asynchronous tracking correctly captures thermodynamic stability!")

if __name__ == "__main__":
    test_async_ksm()
