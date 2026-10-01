"""Dataset 2.5D a nivel de estudio a partir de la caché uint8 [N, S, H, W].

Cada estudio se presenta como P "posiciones" a lo largo del volumen; en cada posición
se apilan el corte y sus dos vecinos como 3 canales (2.5D). Salida: [P, 3, H, W].
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def augment(vol, rng):
    """Misma transformación para todos los cortes del estudio."""
    s, h, w = vol.shape
    if rng.random() < 0.5:           # invertir el orden de cortes ~ rodilla izquierda/derecha
        vol = vol[::-1]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.05, 0.05, size=2) * np.array([w, h])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (tx, ty)
    vol = np.stack([cv2.warpAffine(sl, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0) for sl in vol])
    gamma = rng.uniform(0.8, 1.25)   # contraste no lineal
    vol = np.clip(vol, 0, 1) ** gamma
    vol = vol * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    return np.clip(vol, 0, 1).astype(np.float32)


class KneeStudyDataset(Dataset):
    def __init__(self, volumes, rows, targets, n_pos=16, train=False):
        self.vol = volumes
        self.rows = np.asarray(rows)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.n_pos = n_pos
        self.train = train
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        vol = np.asarray(self.vol[self.rows[i]], dtype=np.float32) / 255.0
        if self.train:
            vol = augment(vol, self.rng)
        s = vol.shape[0]
        pos = np.round(np.linspace(0, s - 1, self.n_pos)).astype(int)
        trip = np.stack([vol[np.clip(pos + d, 0, s - 1)] for d in (-1, 0, 1)], axis=1)  # [P, 3, H, W]
        return torch.from_numpy(np.ascontiguousarray(trip)), torch.from_numpy(self.targets[i])


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(seed + worker_id)
