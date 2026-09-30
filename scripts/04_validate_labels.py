"""Paso 4 — Validación del extractor contra los estudios con etiqueta dorada (gold).

Uso:
    python scripts/04_validate_labels.py

Compara las etiquetas débiles con las 12 etiquetas gold de train.csv y reporta:
  * por etiqueta: acuerdo, precisión, recall, F1 y kappa de Cohen, con dos criterios
    para los casos inciertos ('unc' = 0 y 'unc' = 1);
  * tasa de ruido estimada (1 - acuerdo);
  * acuerdo por idioma (inglés original vs. traducido).
Solo imprime y guarda métricas e identificadores; nunca texto de reportes.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import DEFAULT_DATA_DIR, resolve_data_dir  # noqa: E402
from knee.labels import LABELS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Validación de etiquetas débiles vs. gold")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--weak", default="outputs/labels/weak_labels.csv")
    p.add_argument("--out_dir", default="outputs/labels")
    return p.parse_args()


def metrics(y, p):
    both_const = len(set(y)) == 1 and len(set(p)) == 1
    return {
        "acuerdo": float(np.mean(y == p)),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "kappa": float("nan") if both_const else cohen_kappa_score(y, p),
    }


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["StudyInstanceUID"] + LABELS)
    gold = train[train[LABELS].notna().all(axis=1)]
    weak = pd.read_csv(args.weak)
    df = gold.merge(weak, on="StudyInstanceUID", suffixes=("_gold", ""))
    print(f"Estudios gold con etiqueta débil: {len(df)} / {len(gold)}")

    rows, disagreements = [], []
    for lab in LABELS:
        y = df[f"{lab}_gold"].astype(int).values
        status = df[f"{lab}__status"].values
        p0 = (status == "pos").astype(int)                      # unc -> 0
        p1 = np.isin(status, ["pos", "unc"]).astype(int)        # unc -> 1
        m0, m1 = metrics(y, p0), metrics(y, p1)
        counts = pd.Series(status).value_counts().to_dict()
        rows.append({"etiqueta": lab, "pos_gold": int(y.sum()), "pos_extr": int(p0.sum()),
                     **{f"{k}": round(v, 3) for k, v in m0.items()},
                     "f1_unc=1": round(m1["f1"], 3), "kappa_unc=1": round(m1["kappa"], 3),
                     "n_pos": counts.get("pos", 0), "n_unc": counts.get("unc", 0),
                     "n_neg": counts.get("neg", 0), "n_nm": counts.get("nm", 0)})
        for uid, lang, g, st in zip(df.StudyInstanceUID, df.lang, y, status):
            if g != int(st == "pos"):
                disagreements.append({"StudyInstanceUID": uid, "lang": lang, "etiqueta": lab,
                                      "gold": g, "extraido": st,
                                      "tipo": "falso negativo" if g == 1 else "falso positivo"})
    res = pd.DataFrame(rows).set_index("etiqueta")
    pd.set_option("display.width", 200)
    print("\n=== Acuerdo por etiqueta (criterio principal: unc = 0) ===")
    print(res.to_string())

    macro = res[["acuerdo", "precision", "recall", "f1", "kappa"]].mean()
    print("\nPromedio macro:", macro.round(3).to_dict())
    print(f"Tasa de ruido estimada (1 - acuerdo medio): {1 - macro['acuerdo']:.3f}")

    # Por idioma: acuerdo sobre todas las celdas (estudio x etiqueta)
    cell = []
    for lab in LABELS:
        cell.append(pd.DataFrame({"lang": df.lang, "ok": df[f"{lab}_gold"].astype(int).values ==
                                  (df[f"{lab}__status"] == "pos").astype(int).values,
                                  "gold_pos": df[f"{lab}_gold"].astype(int).values,
                                  "pred_pos": (df[f"{lab}__status"] == "pos").astype(int).values}))
    cell = pd.concat(cell)
    cell["grupo"] = np.where(cell.lang == "en", "inglés original", "traducido")

    def summarize(g):
        tp = int(((g.gold_pos == 1) & (g.pred_pos == 1)).sum())
        return pd.Series({"estudios": len(g) // len(LABELS), "acuerdo": round(g.ok.mean(), 3),
                          "recall": round(tp / max(int(g.gold_pos.sum()), 1), 3),
                          "precision": round(tp / max(int(g.pred_pos.sum()), 1), 3)})

    by_group = cell.groupby("grupo").apply(summarize)
    by_lang = cell.groupby("lang").apply(summarize).sort_values("estudios", ascending=False)
    print("\n=== Acuerdo: inglés original vs. traducido ===")
    print(by_group.to_string())
    print("\n=== Acuerdo por idioma (ojo: pocos estudios gold en varios idiomas) ===")
    print(by_lang.to_string())

    dis = pd.DataFrame(disagreements)
    if len(dis):
        print("\n=== Desacuerdos por tipo y etiqueta ===")
        print(dis.pivot_table(index="etiqueta", columns="tipo", values="gold",
                              aggfunc="count", fill_value=0).to_string())
        print("\nEstado asignado en los falsos negativos:",
              dis[dis.tipo == "falso negativo"].extraido.value_counts().to_dict())

    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, "validation_per_label.csv"))
    by_lang.to_csv(os.path.join(args.out_dir, "validation_per_language.csv"))
    dis.to_csv(os.path.join(args.out_dir, "gold_disagreements.csv"), index=False)
    with open(os.path.join(args.out_dir, "validation_summary.json"), "w") as f:
        json.dump({"macro": macro.round(4).to_dict(), "noise_rate": round(1 - macro["acuerdo"], 4),
                   "by_group": by_group.to_dict(orient="index")}, f, indent=2, default=float)
    print(f"\nArchivos guardados en {args.out_dir}/ "
          "(gold_disagreements.csv lista los casos a revisar dentro de Kaggle)")


if __name__ == "__main__":
    main()
