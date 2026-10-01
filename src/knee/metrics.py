"""Métricas multietiqueta: AUC-ROC por etiqueta y macro con IC bootstrap, umbrales y P/R/F1."""
import warnings

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def auc_per_label(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = y[:, j]
        out.append(roc_auc_score(yj, p[:, j]) if 0 < yj.sum() < len(yj) else np.nan)
    return np.array(out)


def bootstrap_auc(y, p, n_boot=1000, seed=0):
    """IC 95 % por remuestreo de estudios (percentiles). Devuelve (lo, hi) por etiqueta y macro."""
    rng = np.random.default_rng(seed)
    n = len(y)
    per, macro = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            a = auc_per_label(y[idx], p[idx])
            per.append(a)
            macro.append(np.nanmean(a) if np.isfinite(a).any() else np.nan)
    per = np.array(per)
    lo, hi = np.nanpercentile(per, 2.5, axis=0), np.nanpercentile(per, 97.5, axis=0)
    return lo, hi, (np.nanpercentile(macro, 2.5), np.nanpercentile(macro, 97.5))


def best_f1_thresholds(y, p, grid=np.linspace(0.05, 0.95, 91)):
    """Umbral por etiqueta que maximiza F1 (se elige en validación, nunca en test)."""
    th = []
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            th.append(0.5)
            continue
        f1s = [f1_score(y[:, j], p[:, j] >= t, zero_division=0) for t in grid]
        th.append(float(grid[int(np.argmax(f1s))]))
    return np.array(th)


def prf_at(y, p, th):
    pred = p >= th
    rows = []
    for j in range(y.shape[1]):
        rows.append((precision_score(y[:, j], pred[:, j], zero_division=0),
                     recall_score(y[:, j], pred[:, j], zero_division=0),
                     f1_score(y[:, j], pred[:, j], zero_division=0)))
    return np.array(rows)
