"""Paso 8 — Tabla comparativa de todos los experimentos (modelos y ablaciones).

Uso:
    python scripts/08_summarize.py

Lee outputs/runs/*/metrics.json y, si existen, los de corridas anteriores añadidas como Input
(/kaggle/input/**/runs/*/metrics.json). Si un experimento aparece en ambos, gana el local.
Escribe outputs/runs/summary.csv y summary_per_label.csv.
"""
import argparse
import glob
import json
import os

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import find_file  # noqa: E402
from knee.labels import LABELS  # noqa: E402
from knee.metrics import auc_per_label  # noqa: E402

# (experimento, referencia, qué se compara)
COMPARISONS = [
    ("B_resnet50_bce_soft", "A_scratch_bce_soft", "E2 transfer learning vs. desde cero"),
    ("C_resnet50_focal_soft", "B_resnet50_bce_soft", "E4 focal vs. BCE"),
    ("D_resnet50_bce_hard", "B_resnet50_bce_soft", "E4 etiquetas duras vs. blandas"),
    ("B_resnet50_bce_soft", "E_resnet50_bce_soft_nopw", "E6 con vs. sin pos_weight"),
    ("F_resnet50_focal_soft_cor", "C_resnet50_focal_soft", "E4 coronal vs. sagital"),
]


def paired_bootstrap(y, p1, p2, labels_idx=None, n_boot=2000, seed=0):
    """Diferencia de AUC macro (p1 - p2) remuestreando los MISMOS estudios para ambos modelos."""
    rng = np.random.default_rng(seed)
    cols = labels_idx if labels_idx is not None else np.arange(y.shape[1])
    base = np.nanmean(auc_per_label(y, p1)[cols]) - np.nanmean(auc_per_label(y, p2)[cols])
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a1, a2 = auc_per_label(y[i], p1[i])[cols], auc_per_label(y[i], p2[i])[cols]
        if np.isfinite(a1).any() and np.isfinite(a2).any():
            deltas.append(np.nanmean(a1) - np.nanmean(a2))
    deltas = np.array(deltas)
    return base, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5), float((deltas <= 0).mean())


warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="outputs/runs")
    p.add_argument("--rare", default="Fracture,MCL,Synovitis,Lateral OA",
                   help="Etiquetas raras para comparar la mitigación del desbalance (E6)")
    args = p.parse_args()
    rows, per = [], []
    found = {}
    for path in sorted(glob.glob("/kaggle/input/**/runs/*/metrics.json", recursive=True)):
        found[os.path.basename(os.path.dirname(path))] = path
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.json"))):
        found[os.path.basename(os.path.dirname(path))] = path
    os.makedirs(args.runs_dir, exist_ok=True)
    for name in sorted(found):
        m = json.load(open(found[name]))
        rows.append({"run": m["run"], **m["config"], "best_epoch": m["best_epoch"],
                     "val_macro_auc": round(m["val_macro_auc"], 4),
                     "test_macro_auc": round(m["test_macro_auc"], 4),
                     "ic95": f"{m['test_macro_auc_ci95'][0]:.3f}–{m['test_macro_auc_ci95'][1]:.3f}",
                     "test_macro_f1": round(m["test_macro_f1"], 3)})
        for r in m["per_label"]:
            per.append({"run": m["run"], "label": r["label"], "auc_test": r["auc_test"]})
    if not rows:
        print("No hay experimentos en", args.runs_dir)
        return
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("=== Resumen de experimentos (test = estudios gold) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.runs_dir, "summary.csv"), index=False)
    wide = pd.DataFrame(per).pivot(index="label", columns="run", values="auc_test")
    print("\n=== AUC de test por etiqueta ===")
    print(wide.round(3).to_string())
    wide.to_csv(os.path.join(args.runs_dir, "summary_per_label.csv"))
    rare = [r for r in args.rare.split(",") if r in wide.index]
    if rare:
        print(f"\n=== E6: AUC de test en etiquetas raras ({', '.join(rare)}) ===")
        print(wide.loc[rare].round(3).to_string())
        print("Media:", wide.loc[rare].mean().round(3).to_dict())

    # ---- comparaciones pareadas sobre los mismos 58 estudios gold
    try:
        subset = pd.read_csv(find_file("subset.csv", "outputs/subset/subset.csv"))
    except FileNotFoundError:
        print("\n(subset.csv no disponible: se omiten las comparaciones pareadas)")
        return
    gold = subset[subset.split == "test"][["StudyInstanceUID"] + [f"{l}__gold" for l in LABELS]]
    preds = {}
    for name, path in found.items():
        pt = os.path.join(os.path.dirname(path), "preds_test.csv")
        if os.path.exists(pt):
            preds[name] = gold.merge(pd.read_csv(pt), on="StudyInstanceUID")
    rare_idx = np.array([LABELS.index(r) for r in rare]) if rare else None
    out = []
    print("\n=== Comparaciones pareadas (bootstrap de estudios, 2000 réplicas) ===")
    for run, ref, what in COMPARISONS:
        if run not in preds or ref not in preds:
            continue
        a, b = preds[run], preds[ref]
        y = a[[f"{l}__gold" for l in LABELS]].to_numpy(float)
        p1, p2 = a[LABELS].to_numpy(float), b.set_index("StudyInstanceUID").loc[a.StudyInstanceUID, LABELS].to_numpy(float)
        d, lo, hi, p_le0 = paired_bootstrap(y, p1, p2)
        row = {"comparación": what, "experimento": run, "referencia": ref, "ΔAUC macro": round(d, 4),
               "IC95": f"{lo:+.3f} a {hi:+.3f}", "P(Δ<=0)": round(p_le0, 3)}
        if rare_idx is not None:
            dr, lor, hir, _ = paired_bootstrap(y, p1, p2, rare_idx)
            row.update({"ΔAUC raras": round(dr, 4), "IC95 raras": f"{lor:+.3f} a {hir:+.3f}"})
        out.append(row)
    if out:
        res = pd.DataFrame(out)
        print(res.to_string(index=False))
        print("Si el IC95 incluye 0, la diferencia no es concluyente con 58 estudios.")
        res.to_csv(os.path.join(args.runs_dir, "paired_comparisons.csv"), index=False)


if __name__ == "__main__":
    main()
