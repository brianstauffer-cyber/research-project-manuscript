
import numpy as np

def generate_sample(seq_len: int, spoof: bool = False, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, seq_len)
    pseudo = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.05 * rng.standard_normal(seq_len)
    snr = 30 + 2 * np.sin(2 * np.pi * 1.5 * t) + 0.5 * rng.standard_normal(seq_len)
    doppler = 0.1 * np.sin(2 * np.pi * 5 * t) + 0.02 * rng.standard_normal(seq_len)
    if spoof:
        start = rng.integers(seq_len // 8, seq_len // 2)
        end = min(seq_len, start + rng.integers(seq_len // 8, seq_len // 2))
        pseudo[start:end] += rng.uniform(2.0, 6.0)
        snr[start:end] -= rng.uniform(5.0, 12.0)
        if rng.random() > 0.5:
            doppler += np.linspace(0, rng.uniform(0.5, 2.0), seq_len)
    x = np.stack([pseudo, snr, doppler], axis=-1)  # (seq_len, 3)
    return x

def build_dataset(n_samples=1000, seq_len=256, spoof_frac=0.35, seed=42):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, seq_len, 3), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int32)
    for i in range(n_samples):
        is_spoof = (rng.random() < spoof_frac)
        X[i] = generate_sample(seq_len, spoof=is_spoof, seed=rng.integers(1e9))
        y[i] = int(is_spoof)
    return X, y
