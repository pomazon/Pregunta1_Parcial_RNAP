"""Paso 3 — Extracción de etiquetas débiles a partir de los reportes en inglés.

Uso:
    python scripts/03_extract_labels.py

Entrada : outputs/labels/reports_en.csv (paso 2)
Salida  : outputs/labels/weak_labels.csv, una fila por estudio con, para cada etiqueta:
            <etiqueta>          etiqueta dura 0/1 (solo 'pos' = 1)
            <etiqueta>__status  pos / unc / neg / nm
            <etiqueta>__soft    1.0 / 0.5 / 0.0 / 0.0
            <etiqueta>__conf    confianza 1.0 / 0.3 / 1.0 / 0.7
          Este archivo no contiene texto de reportes.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.extractor import (STATUS_TO_CONF, STATUS_TO_HARD, STATUS_TO_SOFT,  # noqa: E402
                            extract_report)
from knee.labels import LABELS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Extracción de etiquetas débiles")
    p.add_argument("--reports", default="outputs/labels/reports_en.csv")
    p.add_argument("--out", default="outputs/labels/weak_labels.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rep = pd.read_csv(args.reports)
    rep["report_en"] = rep["report_en"].fillna("").astype(str)

    rows = []
    for r in rep.itertuples():
        st = extract_report(r.report_en)
        row = {"StudyInstanceUID": r.StudyInstanceUID, "lang": r.lang}
        for lab in LABELS:
            row[lab] = STATUS_TO_HARD[st[lab]]
            row[f"{lab}__status"] = st[lab]
            row[f"{lab}__soft"] = STATUS_TO_SOFT[st[lab]]
            row[f"{lab}__conf"] = STATUS_TO_CONF[st[lab]]
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    dist = pd.DataFrame({lab: out[f"{lab}__status"].value_counts() for lab in LABELS}).T
    dist = dist.reindex(columns=["pos", "unc", "neg", "nm"]).fillna(0).astype(int)
    dist["prev_pos_%"] = (100 * dist["pos"] / len(out)).round(1)
    dist["prev_pos+unc_%"] = (100 * (dist["pos"] + dist["unc"]) / len(out)).round(1)
    print(f"Estudios procesados: {len(out)}")
    print("Distribución de estados por etiqueta (todos los estudios):")
    print(dist.to_string())
    dist.to_csv(os.path.join(os.path.dirname(args.out), "weak_label_distribution.csv"))
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
