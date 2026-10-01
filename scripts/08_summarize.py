"""Paso 8 — Tabla comparativa de todos los experimentos (modelos y ablaciones).

Uso:
    python scripts/08_summarize.py

Lee outputs/runs/*/metrics.json y escribe outputs/runs/summary.csv y summary_per_label.csv.
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="outputs/runs")
    args = p.parse_args()
    rows, per = [], []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.json"))):
        m = json.load(open(path))
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
    print("=== Resumen de experimentos (test = 58 gold) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.runs_dir, "summary.csv"), index=False)
    wide = pd.DataFrame(per).pivot(index="label", columns="run", values="auc_test")
    print("\n=== AUC de test por etiqueta ===")
    print(wide.round(3).to_string())
    wide.to_csv(os.path.join(args.runs_dir, "summary_per_label.csv"))


if __name__ == "__main__":
    main()
