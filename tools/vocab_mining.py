"""Diagnóstico del extractor SIN usar etiquetas (solo estudios NO gold).

Para cada etiqueta busca oraciones que mencionan la estructura o el concepto
(ancla) pero en las que el extractor devolvió 'nm' (no mencionado). Cuenta las
palabras y bigramas que rodean al ancla en esas oraciones: son candidatos a
vocabulario que las reglas no cubren. Separa inglés original y traducido.

No usa los 58 estudios gold, así que mejorar las reglas con esta salida no
contamina la validación. Solo imprime n-gramas frecuentes (>= --min_count
apariciones en estudios distintos), nunca oraciones completas.

Uso:
    python tools/vocab_mining.py
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import DEFAULT_DATA_DIR, resolve_data_dir  # noqa: E402
from knee.extractor import extract_sentence, split_sentences  # noqa: E402
from knee.labels import LABELS  # noqa: E402

I = re.IGNORECASE
ANCHORS = {
    "ACL": r"\bcruciate|\bACL\b",
    "MCL": r"\bcollateral|\bMCL\b|\binternal lateral|\bmedial (?:lateral )?ligament|\binner ligament",
    "Medial Meniscus": r"\bmenisc\w*",
    "Lateral Meniscus": r"\bmenisc\w*",
    "Medial OA": r"\bcartilag\w*|\bchondr\w*|\bosteoarth\w*|\barthros\w*|\bdegenerat\w*",
    "Lateral OA": r"\bcartilag\w*|\bchondr\w*|\bosteoarth\w*|\barthros\w*|\bdegenerat\w*",
    "PF OA": r"\bpatell\w*|\btrochle\w*|\bfemoropatell\w*",
    "Effusion": r"\bfluid|\bliquid|\beffusion|\bhydrops|\brecess|\bsuprapatellar",
    "Synovitis": r"\bsynovi\w*|\bhoffa\w*|\bsynovial",
    "Baker's": r"\bpopliteal|\bbaker|\bgastrocnemi\w*|\bsemimembran\w*",
    "Contusion": r"\bmarrow|\bo?edema|\bbone bruis\w*|\bcontus\w*",
    "Fracture": r"\bfractur\w*|\bfissur\w*|\bbreak\w*|\bcrack\w*|\bfragment\w*",
}
STOP = set("""a an the of in on at to and or is are was were be been with without for from by as
this that these those it its there which who whom whose into within than then also both
signal image images seen shows show shown noted visible appears appear knee right left
""".split())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--reports", default="outputs/labels/reports_en.csv")
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--min_count", type=int, default=15)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out", default="outputs/labels/vocab_mining.txt")
    return p.parse_args()


def tokens(text):
    return re.findall(r"[a-z][a-z\-']+", text.lower())


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["StudyInstanceUID"] + LABELS)
    gold_ids = set(train.loc[train[LABELS].notna().all(axis=1), "StudyInstanceUID"])
    rep = pd.read_csv(args.reports)
    rep = rep[~rep.StudyInstanceUID.isin(gold_ids)]
    rep["report_en"] = rep["report_en"].fillna("").astype(str)
    rep["grupo"] = rep["lang"].eq("en").map({True: "en", False: "trad"})
    print(f"Estudios NO gold analizados: {len(rep)} (gold excluidos: {len(gold_ids)})")

    anchors = {lab: re.compile(rx, I) for lab, rx in ANCHORS.items()}
    # (etiqueta, grupo) -> Counter de n-gramas; contamos 1 vez por estudio
    grams = defaultdict(Counter)
    n_anchor_nm = Counter()
    n_anchor = Counter()
    for r in rep.itertuples():
        seen = defaultdict(set)
        for sent in split_sentences(r.report_en):
            status = None
            for lab, rx in anchors.items():
                m = rx.search(sent)
                if not m:
                    continue
                if status is None:
                    status = extract_sentence(sent)
                n_anchor[(lab, r.grupo)] += 1
                if status[lab] != "nm":
                    continue
                n_anchor_nm[(lab, r.grupo)] += 1
                left = tokens(sent[:m.start()])[-args.window:]
                right = tokens(sent[m.end():])[:args.window]
                ctx = [w for w in left + right if w not in STOP]
                seen[lab].update(ctx)
                seen[lab].update(f"{a} {b}" for a, b in zip(left + right, (left + right)[1:])
                                 if a not in STOP or b not in STOP)
        for lab, g in seen.items():
            grams[(lab, r.grupo)].update(g)

    lines = []
    for lab in LABELS:
        for grupo in ("en", "trad"):
            tot, nm = n_anchor[(lab, grupo)], n_anchor_nm[(lab, grupo)]
            top = [(g, c) for g, c in grams[(lab, grupo)].most_common() if c >= args.min_count][:args.top]
            lines.append(f"\n### {lab} [{grupo}] oraciones con ancla: {tot} | sin detección (nm): {nm} "
                         f"({100 * nm / max(tot, 1):.0f}%)")
            lines.append(", ".join(f"{g}({c})" for g, c in top) or "(nada frecuente)")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)

    # Tasa de 'nm' por idioma y etiqueta (sin etiquetas gold)
    wl = pd.read_csv("outputs/labels/weak_labels.csv")
    wl = wl[~wl.StudyInstanceUID.isin(gold_ids)]
    nm_rate = pd.DataFrame({lab: wl.groupby("lang")[f"{lab}__status"].apply(lambda s: (s == "nm").mean())
                            for lab in LABELS}).round(2)
    nm_rate.insert(0, "estudios", wl.groupby("lang").size())
    print("\n=== Proporción de 'no mencionado' por idioma (estudios NO gold) ===")
    print(nm_rate.sort_values("estudios", ascending=False).to_string())
    nm_rate.to_csv(os.path.join(os.path.dirname(args.out), "nm_rate_by_language.csv"))


if __name__ == "__main__":
    main()
