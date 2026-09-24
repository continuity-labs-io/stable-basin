import numpy as np

def zscore_fit(trajs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    allx = np.concatenate(trajs, axis=0)
    return allx.mean(axis=0), allx.std(axis=0, ddof=1)  # ddof=1 matches torch.std

def stratified_split(labels: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    idx_a, idx_b = [], []
    for lab in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == lab))
        cut = int(round(frac * len(idx)))
        idx_a.extend(idx[:cut])
        idx_b.extend(idx[cut:])
    return np.sort(idx_a), np.sort(idx_b)

def window_starts(T: int, seq_len: int, k: int) -> np.ndarray:
    if T < seq_len:
        raise ValueError(f"Series length {T} < seq_len {seq_len}.")
    return np.unique(np.linspace(0, T - seq_len, k).astype(int))
