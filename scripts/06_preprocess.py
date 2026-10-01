"""Paso 6 — Preprocesamiento DICOM -> caché .npy (una sola vez).

Uso:
    python scripts/06_preprocess.py                 # sagital y coronal
    python scripts/06_preprocess.py --planes Sagittal

Por cada estudio del subconjunto y cada plano:
  1. Lee las cabeceras de la serie y ordena los cortes por posición anatómica
     (ImagePositionPatient proyectada sobre la normal del corte; si falta, InstanceNumber).
  2. Toma --n_slices cortes equiespaciados entre el 5 % y el 95 % del volumen: así se
     cubren LCA (centro) y ambos meniscos (extremos) y se evitan los cortes vacíos del borde.
  3. Decodifica solo esos cortes, aplica RescaleSlope/Intercept, normaliza la serie
     recortando percentiles 1–99 y escala a uint8.
  4. Rellena a cuadrado (sin deformar la anatomía) y redimensiona a --size x --size.

Salida (outputs/cache/):
  <plane>.npy        uint8 [N, n_slices, size, size]
  <plane>_index.csv  fila del .npy por estudio + metadatos
  qc_<plane>.png     mosaico de control visual (NO compartir fuera de Kaggle)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import DEFAULT_DATA_DIR, find_file, resolve_data_dir  # noqa: E402

PLANE_COLS = {"Sagittal": ("sag_series", "sag_n"), "Coronal": ("cor_series", "cor_n")}


def parse_args():
    p = argparse.ArgumentParser(description="Preprocesamiento DICOM a caché")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--subset", default="outputs/subset/subset.csv")
    p.add_argument("--planes", default="Sagittal,Coronal")
    p.add_argument("--n_slices", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--out_dir", default="outputs/cache")
    p.add_argument("--limit", type=int, default=0, help="Procesar solo N estudios (prueba)")
    return p.parse_args()


def _sort_key_fn(headers):
    try:
        iop = np.asarray(headers[0].ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
        keys = [float(np.dot(normal, np.asarray(h.ImagePositionPatient, float))) for h in headers]
        return keys, "position"
    except Exception:
        return [int(getattr(h, "InstanceNumber", i)) for i, h in enumerate(headers)], "instance"


def load_series(series_dir, n_slices, size):
    """Devuelve (volumen uint8 [n_slices, size, size], metadatos)."""
    files = sorted(f for f in os.listdir(series_dir) if f.endswith(".dcm"))
    headers = [pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True) for f in files]
    # descartar cortes con tamaño distinto al mayoritario (localizadores, etc.)
    shapes = [(int(h.Rows), int(h.Columns)) for h in headers]
    main_shape = max(set(shapes), key=shapes.count)
    keep = [i for i, s in enumerate(shapes) if s == main_shape]
    files, headers = [files[i] for i in keep], [headers[i] for i in keep]
    keys, method = _sort_key_fn(headers)
    order = np.argsort(keys)
    n = len(order)
    idx = np.round(np.linspace(0.05 * (n - 1), 0.95 * (n - 1), n_slices)).astype(int)

    slices = []
    for i in idx:
        ds = pydicom.dcmread(os.path.join(series_dir, files[order[i]]))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        slices.append(arr)
    vol = np.stack(slices)
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    h, w = vol.shape[1:]
    m = max(h, w)
    pad = ((0, 0), ((m - h) // 2, m - h - (m - h) // 2), ((m - w) // 2, m - w - (m - w) // 2))
    vol = np.pad(vol, pad)
    out = np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in vol])
    out = (out * 255).round().astype(np.uint8)
    sp = getattr(headers[0], "PixelSpacing", [np.nan, np.nan])
    meta = {"n_orig": n, "rows": h, "cols": w, "sort": method,
            "pixel_spacing": float(sp[0]) if sp is not None else np.nan,
            "slice_thickness": float(getattr(headers[0], "SliceThickness", np.nan) or np.nan)}
    return out, meta


def _job(args):
    series_dir, n_slices, size = args
    try:
        vol, meta = load_series(series_dir, n_slices, size)
        return vol, meta, ""
    except Exception as e:  # se registra y se sigue con el resto
        return None, {}, f"{type(e).__name__}: {str(e)[:120]}"


def save_qc(arr, index, path, n_studies=4, n_cols=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = index[index.ok & (index.split == "train")].head(n_studies)
    if rows.empty:
        return
    cols_idx = np.linspace(0, arr.shape[1] - 1, n_cols).round().astype(int)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2 * n_cols, 2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows.itertuples()):
        for c, k in enumerate(cols_idx):
            axes[r, c].imshow(arr[row.row, k], cmap="gray")
            axes[r, c].set_title(f"corte {k}", fontsize=7)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    subset = pd.read_csv(find_file("subset.csv", args.subset))
    if args.limit:
        subset = subset.groupby("split", group_keys=False).head(max(1, args.limit // 3))
    os.makedirs(args.out_dir, exist_ok=True)

    for plane in [p.strip() for p in args.planes.split(",") if p.strip()]:
        s_col, _ = PLANE_COLS[plane]
        rows = subset[subset[s_col].notna()].reset_index(drop=True)
        print(f"\n== {plane}: {len(rows)} estudios ({rows.split.value_counts().to_dict()}) ==")
        npy_path = os.path.join(args.out_dir, f"{plane}.npy")
        arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8,
                                        shape=(len(rows), args.n_slices, args.size, args.size))
        jobs = [(os.path.join(data_dir, "train_series", u, s), args.n_slices, args.size)
                for u, s in zip(rows.StudyInstanceUID, rows[s_col])]
        metas, errors, t0 = [], [], time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (vol, meta, err) in enumerate(ex.map(_job, jobs, chunksize=4)):
                if vol is not None:
                    arr[i] = vol
                metas.append(meta)
                errors.append(err)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(rows)} | {(time.time() - t0) / 60:.1f} min", flush=True)
        arr.flush()
        index = pd.concat([rows[["StudyInstanceUID", "split", s_col]].rename(columns={s_col: "series"}),
                           pd.DataFrame(metas)], axis=1)
        index["row"] = np.arange(len(rows))
        index["error"] = errors
        index["ok"] = index.error.eq("")
        index.to_csv(os.path.join(args.out_dir, f"{plane}_index.csv"), index=False)
        save_qc(arr, index, os.path.join(args.out_dir, f"qc_{plane}.png"))

        print(f"Correctos: {int(index.ok.sum())}/{len(index)} | tiempo {(time.time() - t0) / 60:.1f} min | "
              f"tamaño {os.path.getsize(npy_path) / 1e6:.0f} MB")
        if (~index.ok).any():
            print("Errores:", index.loc[~index.ok, "error"].value_counts().head(5).to_dict())
        ok = index[index.ok]
        print("Cortes originales por serie:", ok.n_orig.describe().round(1).to_dict())
        print("Resolución original (filas x columnas):",
              (ok.rows.astype(int).astype(str) + "x" + ok.cols.astype(int).astype(str)).value_counts().head(5).to_dict())
        print("Ordenación por:", ok["sort"].value_counts().to_dict())
        print("Espaciado de píxel (mm):", ok.pixel_spacing.describe().round(2).to_dict())


if __name__ == "__main__":
    main()
