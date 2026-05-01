"""Scaler fitting utility."""

import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from ..data import DeepfakeDataset


def fit_scaler(root_dir, img_size, save_path,
               n_samples=2000):
    print("[Scaler] Fitting on training subset ...")
    ds  = DeepfakeDataset(root_dir, 'train',
                          img_size, scaler=None)
    idx = np.random.choice(
        len(ds), min(n_samples, len(ds)), replace=False)
    feats = []
    for i in tqdm(idx, desc='  extracting'):
        _, phys, _ = ds[i]
        feats.append(phys.numpy())
    sc = StandardScaler()
    sc.fit(np.array(feats))
    joblib.dump(sc, save_path)
    print(f"[Scaler] Saved → {save_path}")
    return sc
