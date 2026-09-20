import torch
from torch.utils.data import Dataset


class JAXDictDataset(Dataset):
    """
    A Dataset wrapper that provides random noise initialization for internal states
    alongside the true sensory observations. This is generally used for initializing
    state inference in predictive coding graphs or other energy-based models.
    """

    def __init__(self, base_dataset, d_state):
        self.base = base_dataset
        self.d_state = d_state

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s_true = self.base[idx]
        x_init = torch.randn(self.d_state) * 0.01
        return {"s_true": s_true, "x_init": x_init}
