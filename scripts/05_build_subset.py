"""Paso 5 — Selección del subconjunto de estudios y partición train / val / test.

Uso:
    python scripts/05_build_subset.py

* test  : los estudios gold (58) que tienen serie sagital. Solo para evaluación final.
* train / val : N estudios NO gold con alguna serie sagital (se prefiere fluid-sensitive;
  ver --fluid_only), elegidos para que
  cada etiqueta tenga al menos --min_pos positivos débiles; el resto se completa al azar.
  La validación interna se separa POR HUELLA DE ESCÁNER (fabricante + modelo + software +
  frecuencia + bobina), para no medir "reconocer el escáner" en lugar de la rodilla.

Entradas : train.csv, train_series.csv, cabeceras DICOM (1 archivo por serie elegida)
           outputs/labels/weak_labels.csv (paso 3; o la salida de otro notebook como Input)
Salida   : outputs/subset/subset.csv (sin texto de reportes)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import DEFAULT_DATA_DIR, find_file, resolve_data_dir  # noqa: E402
from knee.labels import LABELS  # noqa: E402

FP_TAGS = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions", "ImagingFrequency", "ReceiveCoilName"]


def parse_args():
    p = argparse.ArgumentParser(description="Subconjunto y particiones")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--weak", default="outputs/labels/weak_labels.csv")
    p.add_argument("--n_studies", type=int, default=600, help="Estudios NO gold (train + val)")
    p.add_argument("--min_pos", type=int, default=40, help="Positivos débiles mínimos por etiqueta")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fluid_only", action="store_true",
                   help="Excluir estudios sin sagital fluid-sensitive (sesga contra centros con PD sin supresión grasa)")
    p.add_argument("--out_dir", default="outputs/subset")
    return p.parse_args()


def pick_series(series_df, data_dir, study, plane, fluid_only=True):
    """Serie del plano pedido; prefiere fluid-sensitive y 15–60 cortes (2D multicorte)."""
    cand = series_df[(series_df.StudyInstanceUID == study) & (series_df.Anatomical_Plane == plane)]
    if fluid_only and (cand.Fluid_Sensitive == 1).any():
        cand = cand[cand.Fluid_Sensitive == 1]
    elif fluid_only:
        return None, 0, False
    best, best_key = None, None
    for sid, fs in zip(cand.SeriesInstanceUID, cand.Fluid_Sensitive):
        d = os.path.join(data_dir, "train_series", study, sid)
        n = len([f for f in os.listdir(d) if f.endswith(".dcm")]) if os.path.isdir(d) else 0
        if n == 0:
            continue
        key = (int(fs == 1), int(15 <= n <= 60), -abs(n - 30))
        if best_key is None or key > best_key:
            best, best_key = (sid, n, bool(fs == 1)), key
    return best if best else (None, 0, False)


def fingerprint(data_dir, study, sid):
    d = os.path.join(data_dir, "train_series", study, sid)
    f = sorted(x for x in os.listdir(d) if x.endswith(".dcm"))[0]
    ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
    vals = []
    for tag in FP_TAGS:
        v = getattr(ds, tag, "")
        if tag == "ImagingFrequency" and v != "":
            v = f"{float(v):.0f}"
        vals.append(str(v).strip())
    return "|".join(vals)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    weak = pd.read_csv(find_file("weak_labels.csv", args.weak))
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["StudyInstanceUID"] + LABELS)
    series = pd.read_csv(os.path.join(data_dir, "train_series.csv"))

    gold_ids = set(train.loc[train[LABELS].notna().all(axis=1), "StudyInstanceUID"])
    sag = series[series.Anatomical_Plane == "Sagittal"]
    if args.fluid_only:
        sag = sag[sag.Fluid_Sensitive == 1]
    has_sag = set(sag.StudyInstanceUID)
    pool = weak[~weak.StudyInstanceUID.isin(gold_ids) & weak.StudyInstanceUID.isin(has_sag)].copy()
    has_fs = set(series[(series.Anatomical_Plane == "Sagittal") & (series.Fluid_Sensitive == 1)].StudyInstanceUID)
    pool["_fs"] = pool.StudyInstanceUID.isin(has_fs)
    print(f"Pool NO gold con serie sagital: {len(pool)} estudios "
          f"({int(pool._fs.sum())} con sagital fluid-sensitive)")
    print("Proporción del pool SIN sagital fluid-sensitive por idioma:",
          (1 - pool.groupby("lang")._fs.mean()).round(2).to_dict())

    # --- selección estratificada: etiquetas de más rara a más común
    is_pos = {lab: pool[f"{lab}__status"].eq("pos").values for lab in LABELS}
    order = sorted(LABELS, key=lambda lab: is_pos[lab].sum())
    uids = pool.StudyInstanceUID.values
    chosen = np.zeros(len(pool), bool)
    for lab in order:
        need = args.min_pos - int((chosen & is_pos[lab]).sum())
        cand = np.flatnonzero(~chosen & is_pos[lab])
        if need > 0 and len(cand):
            chosen[rng.choice(cand, size=min(need, len(cand)), replace=False)] = True
    n_strat = int(chosen.sum())
    rest = np.flatnonzero(~chosen)
    n_fill = max(0, args.n_studies - n_strat)
    chosen[rng.choice(rest, size=min(n_fill, len(rest)), replace=False)] = True
    sel = pool[chosen].copy()
    print(f"Seleccionados: {len(sel)} ({n_strat} por estratificación + {len(sel) - n_strat} al azar)")

    # --- gold como test (los que tengan alguna serie sagital)
    gold = weak[weak.StudyInstanceUID.isin(gold_ids)].copy()
    sel["split"] = "trainval"
    gold["split"] = "test"
    df = pd.concat([sel, gold], ignore_index=True)

    # --- series sagital y coronal + huella de escáner
    rows = []
    for uid, split in zip(df.StudyInstanceUID, df.split):
        sag = pick_series(series, data_dir, uid, "Sagittal", fluid_only=True)
        sag_any = sag if sag[0] else pick_series(series, data_dir, uid, "Sagittal", fluid_only=False)
        cor = pick_series(series, data_dir, uid, "Coronal", fluid_only=True)
        fp = fingerprint(data_dir, uid, sag_any[0]) if sag_any[0] else ""
        rows.append({"sag_series": sag_any[0], "sag_n": sag_any[1], "sag_fluid": sag_any[2],
                     "cor_series": cor[0], "cor_n": cor[1], "fingerprint": fp})
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    no_sag = df[df.sag_series.isna()]
    if len(no_sag):
        print(f"AVISO: {len(no_sag)} estudios sin ninguna serie sagital (se excluyen):",
              no_sag.split.value_counts().to_dict())
    df = df[df.sag_series.notna()].copy()

    # --- validación interna agrupada por huella de escáner
    tv = df[df.split == "trainval"]
    groups = tv.fingerprint.value_counts()
    print(f"Huellas de escáner distintas en train/val: {len(groups)} "
          f"(la mayor cubre {groups.iloc[0]} estudios)")
    # Los escáneres cuya sagital no es fluid-sensitive (casi todo el centro alemán) van
    # siempre a train: si cayeran en val, el modelo no vería nunca ese protocolo.
    fs_ratio = tv.groupby("fingerprint").sag_fluid.mean()
    train_only = set(fs_ratio[fs_ratio < 0.5].index)
    print(f"Huellas forzadas a train por sagital no fluid-sensitive: {len(train_only)} "
          f"({int(tv.fingerprint.isin(train_only).sum())} estudios)")
    target = args.val_frac * len(tv)
    val_groups, n_val = set(), 0
    for g in rng.permutation([g for g in groups.index.values if g not in train_only]):
        if n_val >= target:
            break
        if n_val + groups[g] > 1.5 * target:
            continue
        val_groups.add(g)
        n_val += groups[g]
    df.loc[(df.split == "trainval") & df.fingerprint.isin(val_groups), "split"] = "val"
    df.loc[df.split == "trainval", "split"] = "train"

    # --- etiquetas gold (solo test)
    gold_lab = train[train.StudyInstanceUID.isin(gold_ids)].rename(columns={l: f"{l}__gold" for l in LABELS})
    df = df.merge(gold_lab, on="StudyInstanceUID", how="left")

    os.makedirs(args.out_dir, exist_ok=True)
    keep = (["StudyInstanceUID", "split", "lang", "fingerprint", "sag_series", "sag_n", "sag_fluid",
             "cor_series", "cor_n"] + [c for l in LABELS for c in (l, f"{l}__soft", f"{l}__status", f"{l}__gold")])
    df[keep].to_csv(os.path.join(args.out_dir, "subset.csv"), index=False)

    # --- resumen (E6: prevalencias)
    print("\nEstudios por partición:", df.split.value_counts().to_dict())
    print("Con coronal fluid-sensitive:", df.groupby("split").cor_series.apply(lambda s: int(s.notna().sum())).to_dict())
    print("Sagital no fluid-sensitive (respaldo):", df.groupby("split").sag_fluid.apply(lambda s: int((~s).sum())).to_dict())
    print("Idiomas train/val:", df[df.split != "test"].lang.value_counts().to_dict())
    print("Sagital fluid-sensitive por idioma (train/val):")
    print(pd.crosstab(df[df.split != "test"].lang, df[df.split != "test"].sag_fluid).to_string())
    prev = pd.DataFrame({
        "pool_pos_%": [round(100 * pool[f"{l}__status"].eq("pos").mean(), 1) for l in LABELS],
        "train_pos": [int(df[df.split == "train"][l].sum()) for l in LABELS],
        "train_soft": [round(df[df.split == "train"][f"{l}__soft"].sum(), 1) for l in LABELS],
        "train_pos_%": [round(100 * df[df.split == "train"][l].mean(), 1) for l in LABELS],
        "val_pos": [int(df[df.split == "val"][l].sum()) for l in LABELS],
        "test_gold_pos": [int(df[df.split == "test"][f"{l}__gold"].sum()) for l in LABELS],
    }, index=LABELS)
    print("\nPrevalencia por etiqueta (pos = positivos débiles; soft = suma de etiquetas blandas):")
    print(prev.to_string())
    prev.to_csv(os.path.join(args.out_dir, "prevalence.csv"))
    low_val = prev.index[prev.val_pos < 5].tolist()
    if low_val:
        print("AVISO: etiquetas con < 5 positivos en val (umbral poco estable):", low_val)
    print(f"\nGuardado en {args.out_dir}/subset.csv")


if __name__ == "__main__":
    main()
