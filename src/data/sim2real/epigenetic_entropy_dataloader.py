import torch
from torch.utils.data import Dataset

class EpigeneticEntropyLoader(Dataset):
    """
    Epigenetic Entropy Dataloader for the Fedichev framework.
    Mocks CpG methylation arrays across thousands of cells.
    To avoid VRAM explosions, massive tensors are generated dynamically per batch.
    """
    def __init__(self, biological_age: int, size: int = 100, seq_len: int = 50, n_cells: int = 1000, n_cpgs: int = 10000):
        super().__init__()
        self.biological_age = biological_age
        self.size = size
        self.seq_len = seq_len
        self.n_cells = n_cells
        self.n_cpgs = n_cpgs
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Shape: [Time, Cells, CpGs]
        shape = (self.seq_len, self.n_cells, self.n_cpgs)
        
        if self.biological_age <= 45:
            # Low Z (Low Entropy): Tightly clustered bimodal distribution (0.0 or 1.0)
            # Pristine epigenetic landscape where cells are strongly committed to their states.
            # Crucially, the base state is consistent across ALL cells for a given CpG site.
            base_cpgs = torch.randint(0, 2, (1, 1, self.n_cpgs), dtype=torch.float32)
            base = base_cpgs.expand(self.seq_len, self.n_cells, self.n_cpgs)
            noise = torch.randn(shape) * 0.05
            cpg_tensor = torch.clamp(base + noise, 0.0, 1.0)
        else:
            # High Z (High Entropy): Wide Gaussian centered at 0.5
            # Cells have lost their structural integrity and are drifting into chaos
            # The older the biological age, the higher the variance
            noise_scale = 0.3 if self.biological_age >= 50 else 0.15
            cpg_tensor = torch.randn(shape) * noise_scale + 0.5
            cpg_tensor = torch.clamp(cpg_tensor, 0.0, 1.0)
            
        return {"cpg_tensor": cpg_tensor, "biological_age": self.biological_age}

if __name__ == "__main__":
    import time
    start = time.time()
    # Test generation
    dataset_45 = EpigeneticEntropyLoader(biological_age=45, size=1, seq_len=10)
    dataset_50 = EpigeneticEntropyLoader(biological_age=50, size=1, seq_len=10)
    
    t_45 = dataset_45[0]["cpg_tensor"]
    t_50 = dataset_50[0]["cpg_tensor"]
    
    print(f"Age 45 Tensor Shape: {t_45.shape}")
    print(f"Age 50 Tensor Shape: {t_50.shape}")
    print(f"Generation took: {time.time() - start:.2f} seconds")
