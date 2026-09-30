"""Paso 1 — Exploración de datos (EDA) tabular y DICOM.

Uso (en un Kaggle Notebook, con la competencia añadida como Input):
    python scripts/01_eda.py --data_dir /kaggle/input/competitions/rsna-knee-abnormality-detection

Imprime y guarda SOLO estadísticas agregadas. Nunca imprime texto de reportes
(reglas de seguridad de datos de la competencia).
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.labels import LABELS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="EDA del dataset RSNA Knee")
    p.add_argument("--data_dir", default="/kaggle/input/competitions/rsna-knee-abnormality-detection")
    p.add_argument("--out_dir", default="outputs/eda")
    p.add_argument("--n_series_sample", type=int, default=300)
    return p.parse_args()


def resolve_data_dir(path):
    """Devuelve la carpeta que contiene train.csv y train_series.csv.

    Si `path` no existe o no los contiene, los busca bajo /kaggle/input
    (Kaggle puede montar la competencia en rutas distintas).
    """
    if os.path.isfile(os.path.join(path, "train_series.csv")):
        return path
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_series.csv" in files and "train.csv" in files:
            print(f"AVISO: datos no encontrados en {path}; se usan los de {root}")
            return root
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    raise FileNotFoundError(
        f"No se encontró train_series.csv en {path} ni bajo /kaggle/input. "
        "¿Está añadida la competencia como Input?")


def safe_lang(text):
    from langdetect import detect
    try:
        return detect(text) if isinstance(text, str) and len(text.strip()) > 20 else "na"
    except Exception:
        return "err"


def main():
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    print("Carpeta de datos:", args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0

    print("Archivos en la raíz:", sorted(os.listdir(args.data_dir)))
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    series = pd.read_csv(os.path.join(args.data_dir, "train_series.csv"))
    print("train.csv:", train.shape, "| train_series.csv:", series.shape)
    print("Columnas train:", list(train.columns))
    print("Columnas series:", list(series.columns))

    # ---------- 1. Etiquetas doradas ----------
    labels = [c for c in LABELS if c in train.columns]
    missing = [c for c in LABELS if c not in train.columns]
    if missing:
        print("AVISO: columnas de etiqueta esperadas y no encontradas:", missing)
    n_all = train[labels].notna().all(axis=1)
    print(f"\nEstudios con las {len(labels)} etiquetas (gold): {int(n_all.sum())} / {len(train)}")
    gold = train[n_all]
    gold_pos = gold[labels].sum().astype(int)
    print("Positivos por etiqueta en gold:\n" + gold_pos.to_string())
    print("Etiquetas con < 3 positivos en gold:", list(gold_pos[gold_pos < 3].index))

    # ---------- 2. Reportes ----------
    rep = train["Report"].fillna("").astype(str)
    train["rep_len"] = rep.str.len()
    train["lang"] = train["Report"].apply(safe_lang)
    print("\nEstudios con reporte:", int((rep.str.strip() != "").sum()), "/", len(train))
    print("Longitud (caracteres):", train.loc[train.rep_len > 0, "rep_len"].describe().round(0).to_dict())
    print("Idiomas (todos):", train["lang"].value_counts().to_dict())
    print("Idiomas (gold):", train.loc[n_all, "lang"].value_counts().to_dict())

    # ---------- 3. Series ----------
    print("\nPlanos:", series["Anatomical_Plane"].value_counts().to_dict())
    if {"Fluid_Sensitive", "Fat_Suppression"} <= set(series.columns):
        print("¿Fluid_Sensitive == Fat_Suppression en todas las filas?",
              bool((series.Fluid_Sensitive == series.Fat_Suppression).all()))
    print("Series por estudio:", series.groupby("StudyInstanceUID").size().describe().round(1).to_dict())
    fs = series["Fluid_Sensitive"] == 1 if "Fluid_Sensitive" in series.columns else True
    coverage = {}
    for plane in ["Sagittal", "Coronal", "Axial"]:
        sel = series[(series.Anatomical_Plane == plane) & fs]
        n = sel.StudyInstanceUID.nunique()
        ng = sel[sel.StudyInstanceUID.isin(gold.StudyInstanceUID)].StudyInstanceUID.nunique()
        coverage[plane] = {"studies": n, "gold": ng}
        print(f"Estudios con {plane} fluid-sensitive: {n} (gold: {ng}/{len(gold)})")

    # ---------- 4. DICOM (muestra de cabeceras) ----------
    sample = series.sample(min(args.n_series_sample, len(series)), random_state=0)
    n_slices, syntaxes, example = [], Counter(), {}
    for _, r in sample.iterrows():
        d = os.path.join(args.data_dir, "train_series", r.StudyInstanceUID, r.SeriesInstanceUID)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith(".dcm")]
        if not files:
            continue
        n_slices.append(len(files))
        ds = pydicom.dcmread(os.path.join(d, files[0]), stop_before_pixels=True)
        ts = str(ds.file_meta.TransferSyntaxUID.name)
        syntaxes[ts] += 1
        example.setdefault(ts, os.path.join(d, files[0]))
    print("\nCortes por serie (muestra):", pd.Series(n_slices).describe().round(1).to_dict())
    print("Transfer syntaxes:", dict(syntaxes))
    decode = {}
    for ts, path in example.items():
        try:
            arr = pydicom.dcmread(path).pixel_array
            decode[ts] = f"OK {arr.shape}"
        except Exception as e:  # falta un códec
            decode[ts] = f"FALLA {type(e).__name__}"
        print(f"  {ts}: {decode[ts]}")

    # ---------- 5. Guardar ----------
    summary = {
        "n_train": len(train), "n_gold": int(n_all.sum()),
        "gold_positives": gold_pos.to_dict(),
        "langs_all": train["lang"].value_counts().to_dict(),
        "langs_gold": train.loc[n_all, "lang"].value_counts().to_dict(),
        "planes": series["Anatomical_Plane"].value_counts().to_dict(),
        "fluid_sensitive_coverage": coverage,
        "syntaxes": dict(syntaxes), "decode": decode,
    }
    with open(os.path.join(args.out_dir, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=int)
    train[["StudyInstanceUID", "lang", "rep_len"]].to_csv(
        os.path.join(args.out_dir, "study_meta.csv"), index=False)
    print(f"\nResumen guardado en {args.out_dir}/eda_summary.json")


if __name__ == "__main__":
    main()
