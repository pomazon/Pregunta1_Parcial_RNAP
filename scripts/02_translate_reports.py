"""Paso 2 — Traducción de los reportes al inglés con un modelo abierto (NLLB-200).

Los reportes que no están en inglés se traducen dentro de Kaggle con
facebook/nllb-200-distilled-600M, sin enviar texto a ninguna API externa.
Antes de traducir se aplica un glosario clínico (src/knee/glossary.py) que fija
en inglés los términos clave que NLLB suele deformar (desactivar con --no_glossary).
Luego un único extractor de reglas en inglés (paso 3) obtiene las 12 etiquetas.

Uso (Kaggle, GPU T4, Internet activado):
    python scripts/02_translate_reports.py

Salida: outputs/labels/reports_en.csv  (contiene texto: NO subir a GitHub ni compartir)
Si la salida ya existe, el paso se omite (usar --force para rehacerlo).
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import DEFAULT_DATA_DIR, detect_language, resolve_data_dir  # noqa: E402
from knee.glossary import apply_glossary  # noqa: E402

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Código langdetect -> código NLLB. Los idiomas no listados se dejan sin traducir.
LANG2NLLB = {
    "es": "spa_Latn", "tr": "tur_Latn", "hr": "hrv_Latn", "el": "ell_Grek",
    "de": "deu_Latn", "bg": "bul_Cyrl", "nl": "nld_Latn", "fr": "fra_Latn",
    "pt": "por_Latn", "it": "ita_Latn", "sr": "srp_Cyrl", "sl": "slv_Latn",
    "mk": "mkd_Cyrl", "pl": "pol_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl",
    "cs": "ces_Latn", "sk": "slk_Latn", "hu": "hun_Latn", "ca": "cat_Latn",
    "sq": "als_Latn", "bs": "bos_Latn", "uk": "ukr_Cyrl", "sv": "swe_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn",
}
MAX_SEG_CHARS = 350


def parse_args():
    p = argparse.ArgumentParser(description="Traducción de reportes al inglés")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default="outputs/labels")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="Traducir solo N reportes (prueba rápida)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--no_glossary", action="store_true", help="Traducir sin el glosario clínico (v1)")
    return p.parse_args()


def split_segments(line):
    """Divide una línea en oraciones y corta las demasiado largas."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", line) if s.strip()]
    out = []
    for s in sents:
        while len(s) > MAX_SEG_CHARS:
            cut = max(s.rfind(", ", 0, MAX_SEG_CHARS), s.rfind(" ", 0, MAX_SEG_CHARS))
            cut = cut if cut > 50 else MAX_SEG_CHARS
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


def segment_report(text):
    """Lista de líneas; cada línea es una lista de segmentos a traducir."""
    return [split_segments(line) for line in str(text).split("\n")]


def main():
    args = parse_args()
    out_csv = os.path.join(args.out_dir, "reports_en.csv")
    if os.path.exists(out_csv) and not args.force:
        print(f"{out_csv} ya existe: se omite la traducción (usar --force para rehacerla).")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        sys.exit("No hay GPU. Activen Accelerator = GPU T4 en Kaggle (o usen --allow_cpu, muy lento).")

    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["StudyInstanceUID", "Report"])
    if args.limit:
        train = train.head(args.limit)
    train["Report"] = train["Report"].fillna("").astype(str)
    train["lang"] = train["Report"].apply(detect_language)
    print("Idiomas:", train["lang"].value_counts().to_dict())
    if not args.no_glossary:
        is_src = train["lang"].isin(list(LANG2NLLB))
        before = train.loc[is_src, "Report"].copy()
        train.loc[is_src, "Report"] = [apply_glossary(t, l) for t, l in
                                       zip(train.loc[is_src, "Report"], train.loc[is_src, "lang"])]
        changed = (before != train.loc[is_src, "Report"]).groupby(train.loc[is_src, "lang"]).mean()
        print("Glosario aplicado. Proporción de reportes modificados por idioma:",
              changed.round(2).to_dict())

    # Segmentos únicos por idioma (los reportes con plantilla repiten frases)
    to_do = {}
    layouts = {}
    for row in train.itertuples():
        if row.lang in LANG2NLLB:
            layout = segment_report(row.Report)
            layouts[row.StudyInstanceUID] = layout
            for segs in layout:
                to_do.setdefault(row.lang, set()).update(segs)
    n_total = sum(len(v) for v in to_do.values())
    print(f"Segmentos únicos a traducir: {n_total}")

    cache_path = os.path.join(args.out_dir, "translation_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Caché previo: {len(cache)} segmentos ya traducidos")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()
    eng_id = tok.convert_tokens_to_ids("eng_Latn")

    t0, done, n_batches = time.time(), 0, 0
    for lang, segs in sorted(to_do.items(), key=lambda kv: -len(kv[1])):
        pending = sorted((s for s in segs if f"{lang}\t{s}" not in cache), key=len)
        done += len(segs) - len(pending)
        tok.src_lang = LANG2NLLB[lang]
        print(f"[{lang}] {len(pending)} segmentos pendientes")
        for i in range(0, len(pending), args.batch_size):
            batch = pending[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=eng_id, num_beams=args.num_beams,
                                     max_new_tokens=min(400, int(enc["input_ids"].shape[1] * 1.6) + 10))
            for src, tgt in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                cache[f"{lang}\t{src}"] = tgt
            done += len(batch)
            n_batches += 1
            if n_batches % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                el = time.time() - t0
                eta = el / max(done, 1) * (n_total - done)
                print(f"  {done}/{n_total} segmentos | {el/60:.1f} min | ETA {eta/60:.1f} min", flush=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    def rebuild(row):
        if row.lang not in LANG2NLLB:
            return row.Report
        return "\n".join(" ".join(cache[f"{row.lang}\t{s}"] for s in segs)
                         for segs in layouts[row.StudyInstanceUID])

    train["translated"] = train["lang"].isin(list(LANG2NLLB))
    train["glossary"] = train["translated"] & (not args.no_glossary)
    train["report_en"] = train.apply(rebuild, axis=1)
    train[["StudyInstanceUID", "lang", "translated", "glossary", "report_en"]].to_csv(out_csv, index=False)
    untranslated = train.loc[~train.translated & (train.lang != "en"), "lang"].value_counts().to_dict()
    print(f"\nTraducidos: {int(train.translated.sum())} | en inglés: {int((train.lang == 'en').sum())} "
          f"| otros sin traducir: {untranslated}")
    print(f"Tiempo total: {(time.time() - t0)/60:.1f} min · Guardado en {out_csv}")


if __name__ == "__main__":
    main()
