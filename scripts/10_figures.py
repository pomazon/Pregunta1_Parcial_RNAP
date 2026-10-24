"""Paso 10 — Figuras y tablas del informe a partir de los resultados ya calculados.

Uso:
    python scripts/10_figures.py
    python scripts/10_figures.py --best C_resnet50_focal_soft

No entrena nada ni lee imágenes: solo métricas y predicciones. Busca los resultados en
outputs/ (corrida con run_all.sh) y, si faltan, bajo /kaggle/input (salidas de otros
notebooks añadidas como Input: P1_Preprocesado, P1_Entrenamiento, P1_Experimentos2 y,
opcional, P1_Lanzador). Si un experimento aparece en ambos lugares, gana el local.

Salida (outputs/figures/), cada figura en .pdf (para LaTeX) y .png:
  fig1_prevalencia          prevalencia por hallazgo: train y val (débiles) y test (gold)   E6
  fig2_auc_macro            AUC macro de test con IC 95 % por experimento                   E2/E4
  fig3_auc_por_hallazgo     mapa de calor AUC de test, hallazgo x experimento               E3
  fig4_roc_mejor            curvas ROC por hallazgo del mejor modelo                        E3
  fig5_comparaciones        diferencias pareadas de AUC con IC 95 %                         E2/E4/E6
  fig6_entrenamiento        pérdida y AUC de validación por epoch                           E2
  fig7_extractor_idioma     recall del extractor por idioma (si se encuentra)               E1/E7
  tablas.tex                tablas en LaTeX (booktabs) para el informe
Las figuras solo contienen estadísticas agregadas (ninguna imagen ni texto de reportes).
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import find_file  # noqa: E402
from knee.labels import LABELS, LABELS_ES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Experimentos en el orden del informe
RUNS = {
    "A_scratch_bce_soft": "A · CNN propia",
    "B_resnet50_bce_soft": "B · ResNet-50 (BCE)",
    "C_resnet50_focal_soft": "C · ResNet-50 focal",
    "D_resnet50_bce_hard": "D · etiquetas duras",
    "E_resnet50_bce_soft_nopw": "E · sin pos_weight",
    "F_resnet50_focal_soft_cor": "F · coronal (focal)",
}
LANG_ES = {"en": "Inglés", "es": "Español", "tr": "Turco", "hr": "Croata", "el": "Griego",
           "de": "Alemán", "bg": "Búlgaro", "nl": "Neerlandés", "fr": "Francés"}

# Paleta categórica validada (orden fijo) + tinta y rejilla
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
    "savefig.bbox": "tight", "text.color": INK,
})


def parse_args():
    p = argparse.ArgumentParser(description="Figuras y tablas del informe")
    p.add_argument("--runs_dir", default="outputs/runs")
    p.add_argument("--search_root", default="/kaggle/input")
    p.add_argument("--best", default="C_resnet50_focal_soft", help="Experimento para ROC y tabla por hallazgo")
    p.add_argument("--out_dir", default="outputs/figures")
    return p.parse_args()


# ------------------------------------------------------------------ carga
def find_runs(runs_dir, search_root):
    found = {}
    for path in sorted(glob.glob(os.path.join(search_root, "**", "runs", "*", "metrics.json"), recursive=True)):
        found[os.path.basename(os.path.dirname(path))] = os.path.dirname(path)
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        found[os.path.basename(os.path.dirname(path))] = os.path.dirname(path)
    order = [r for r in RUNS if r in found] + sorted(r for r in found if r not in RUNS)
    return {r: found[r] for r in order}


def optional_file(name, local, search_root):
    try:
        return find_file(name, local, search_root)
    except FileNotFoundError:
        print(f"AVISO: no se encontró {name}; se omite lo que depende de él")
        return None


def label_of(run):
    return RUNS.get(run, run)


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ------------------------------------------------------------------ figuras
def fig_prevalence(subset, out_dir):
    tr, va, te = (subset[subset.split == s] for s in ("train", "val", "test"))
    prev = pd.DataFrame({
        f"Entrenamiento, débil (n={len(tr)})": [100 * tr[l].mean() for l in LABELS],
        f"Validación, débil (n={len(va)})": [100 * va[l].mean() for l in LABELS],
        f"Test, gold (n={len(te)})": [100 * te[f"{l}__gold"].mean() for l in LABELS],
    }, index=[LABELS_ES[l] for l in LABELS]).sort_values(f"Entrenamiento, débil (n={len(tr)})")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(prev))
    h = 0.26
    for k, col in enumerate(prev.columns):
        ax.barh(y + (k - 1) * h, prev[col], height=h - 0.03, color=SERIES[k], label=col)
    ax.set_yticks(y, prev.index)
    ax.set_xlabel("Estudios positivos (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.2, handlelength=1.2)
    ax.set_xlim(0, prev.values.max() * 1.08)
    ax.set_title("Prevalencia por hallazgo y partición", loc="left", color=INK, pad=22)
    save(fig, out_dir, "fig1_prevalencia")
    return prev


def fig_macro(metrics, out_dir):
    runs = list(metrics)[::-1]
    auc = [metrics[r]["test_macro_auc"] for r in runs]
    lo = [metrics[r]["test_macro_auc_ci95"][0] for r in runs]
    hi = [metrics[r]["test_macro_auc_ci95"][1] for r in runs]
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(runs) + 0.9))
    y = np.arange(len(runs))
    ax.hlines(y, lo, hi, color=SERIES[0], lw=2)
    ax.plot(auc, y, "o", ms=8, color=SERIES[0], mec="white", mew=1.5)
    for yi, a, h in zip(y, auc, hi):
        ax.text(h + 0.008, yi, f"{a:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=NEUTRAL, lw=1, ls="--")
    ax.text(0.503, -0.45, "azar (0.5)", fontsize=7, color=MUTED, va="center")
    ax.set_yticks(y, [label_of(r) for r in runs])
    ax.set_xlabel("AUC-ROC macro en test (gold), con IC 95 % por bootstrap")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0.45, min(lo) - 0.02), max(hi) + 0.06)
    ax.set_ylim(-0.8, len(runs) - 0.5)
    ax.set_title("AUC macro por experimento", loc="left", color=INK)
    save(fig, out_dir, "fig2_auc_macro")


def fig_heatmap(metrics, out_dir):
    runs = list(metrics)
    mat = np.array([[next((r["auc_test"] for r in metrics[run]["per_label"] if r["label"] == l), np.nan)
                     for run in runs] for l in LABELS], dtype=float)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(1.0 * len(runs) + 2.4, 5.0))
    im = ax.imshow(mat, cmap=cmap, vmin=0.4, vmax=0.9, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v >= 0.72 else INK)
    ax.set_xticks(range(len(runs)), [label_of(r).split(" · ")[0] for r in runs])
    ax.set_yticks(range(len(LABELS)), [LABELS_ES[l] for l in LABELS])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("AUC en test")
    cb.outline.set_visible(False)
    ax.set_title("AUC por hallazgo y experimento", loc="left", color=INK)
    save(fig, out_dir, "fig3_auc_por_hallazgo")


def fig_roc(best, run_dir, subset, metrics, out_dir):
    preds = pd.read_csv(os.path.join(run_dir, "preds_test.csv"))
    df = subset[subset.split == "test"].merge(preds, on="StudyInstanceUID", suffixes=("", "__p"))
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for ax, lab in zip(axes.flat, LABELS):
        y = df[f"{lab}__gold"].astype(int).values
        p = df[f"{lab}__p"].values if f"{lab}__p" in df else df[lab].values
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls="--")
        auc = next((r["auc_test"] for r in metrics[best]["per_label"] if r["label"] == lab), np.nan)
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=SERIES[0], lw=2)
        ax.set_title(f"{LABELS_ES[lab]}\nAUC {auc:.2f} · {int(y.sum())} pos.", fontsize=7.5, color=INK)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("1 − especificidad", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Sensibilidad", fontsize=7.5)
    fig.suptitle(f"Curvas ROC en test (gold) — {label_of(best)}", x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out_dir, "fig4_roc_mejor")


def _ci(text):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(text))
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else (np.nan, np.nan)


def fig_comparisons(paired, out_dir):
    df = paired.iloc[::-1].reset_index(drop=True)
    d = df["ΔAUC macro"].astype(float).values
    ci = np.array([_ci(t) for t in df["IC95"]])
    sig = (ci[:, 0] > 0) | (ci[:, 1] < 0)
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(df) + 1.0))
    y = np.arange(len(df))
    for yi, di, (lo, hi), s in zip(y, d, ci, sig):
        c = SERIES[0] if s else NEUTRAL
        ax.hlines(yi, lo, hi, color=c, lw=2)
        ax.plot(di, yi, "o", ms=8, color=c if s else "white", mec=c, mew=1.8)
        ax.text(hi + 0.006, yi, f"{di:+.3f}" + ("  significativa" if s else ""), va="center",
                fontsize=7.5, color=INK if s else MUTED)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [re.sub(r"^E\d+:?\s*", "", c) for c in df["comparación"]])
    ax.set_xlabel("Diferencia de AUC macro (bootstrap pareado, IC 95 %)")
    ax.grid(axis="y", visible=False)
    lim = np.nanmax(np.abs(ci)) + 0.07
    ax.set_xlim(-lim, lim + 0.05)
    ax.set_title("Comparaciones pareadas sobre los mismos estudios gold", loc="left", color=INK)
    save(fig, out_dir, "fig5_comparaciones")


def fig_history(histories, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for k, (run, h) in enumerate(histories.items()):
        c = SERIES[k % len(SERIES)]
        a1.plot(h.epoch, h.train_loss, color=c, lw=2, label=label_of(run))
        a2.plot(h.epoch, h.val_macro_auc, color=c, lw=2, label=label_of(run))
        b = h.val_macro_auc.idxmax()
        a2.plot(h.epoch[b], h.val_macro_auc[b], "o", ms=7, color=c, mec="white", mew=1.5)
    a1.set_title("Pérdida de entrenamiento", loc="left", color=INK)
    a1.set_xlabel("Epoch")
    a1.text(0.98, 0.95, "BCE y focal no están\nen la misma escala", transform=a1.transAxes,
            ha="right", va="top", fontsize=6.8, color=MUTED)
    a2.set_title("AUC macro en validación (● = mejor epoch)", loc="left", color=INK)
    a2.set_xlabel("Epoch")
    a2.legend(fontsize=6.8, loc="lower right")
    from matplotlib.ticker import MaxNLocator
    for a in (a1, a2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, out_dir, "fig6_entrenamiento")


def fig_language(val_lang, out_dir):
    df = val_lang.copy()
    df.index = [LANG_ES.get(i, i) for i in df.index]
    df = df.sort_values("recall")
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(df) + 1.0))
    y = np.arange(len(df))
    colors = [SERIES[1] if i == "Inglés" else SERIES[0] for i in df.index]
    ax.barh(y, df.recall, height=0.6, color=colors)
    for yi, r, n in zip(y, df.recall, df.estudios):
        ax.text(r + 0.01, yi, f"{r:.2f}  (n={int(n)})", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, df.index)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Recall del extractor frente a gold (inglés original en naranja; traducidos en azul)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Calidad de las etiquetas débiles por idioma del informe", loc="left", color=INK)
    save(fig, out_dir, "fig7_extractor_idioma")


# ------------------------------------------------------------------ tablas LaTeX
def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tables_tex(metrics, best, paired, prev):
    out = []
    out.append("% Generado por scripts/10_figures.py — requiere \\usepackage{booktabs}\n")
    out.append("\\begin{table}[t]\\centering\\small\n\\caption{Resultados en test (58 estudios gold; F: 56 con coronal).}\n"
               "\\label{tab:resumen}\n\\begin{tabular}{lcccc}\n\\toprule\n"
               "Experimento & AUC val & AUC test & IC 95\\,\\% & F1 test \\\\\n\\midrule\n")
    for r, m in metrics.items():
        lo, hi = m["test_macro_auc_ci95"]
        out.append(f"{_tex(label_of(r))} & {m['val_macro_auc']:.3f} & {m['test_macro_auc']:.3f} & "
                   f"{lo:.3f}--{hi:.3f} & {m['test_macro_f1']:.3f} \\\\\n")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    out.append("\\begin{table}[t]\\centering\\small\n"
               f"\\caption{{Desglose por hallazgo en test, {_tex(label_of(best))}. Umbral elegido en validación (máximo F1).}}\n"
               "\\label{tab:hallazgos}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
               "Hallazgo & Pos. & AUC [IC 95\\,\\%] & Umbral & Precisión & Recall & F1 \\\\\n\\midrule\n")
    for r in metrics[best]["per_label"]:
        out.append(f"{_tex(LABELS_ES.get(r['label'], r['label']))} & {int(r['pos_test'])} & "
                   f"{r['auc_test']:.2f} [{r['ic95_lo']:.2f}--{r['ic95_hi']:.2f}] & {r['umbral_val']:.2f} & "
                   f"{r['precision']:.2f} & {r['recall']:.2f} & {r['f1']:.2f} \\\\\n")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    if paired is not None:
        out.append("\\begin{table}[t]\\centering\\small\n\\caption{Comparaciones pareadas (bootstrap de 2000 "
                   "réplicas sobre los mismos estudios gold).}\n\\label{tab:pareadas}\n"
                   "\\begin{tabular}{lccc}\n\\toprule\nComparación & $\\Delta$AUC & IC 95\\,\\% & "
                   "$P(\\Delta\\le 0)$ \\\\\n\\midrule\n")
        for _, r in paired.iterrows():
            out.append(f"{_tex(r['comparación'])} & {float(r['ΔAUC macro']):+.3f} & {_tex(r['IC95'])} & "
                       f"{float(r['P(Δ<=0)']):.3f} \\\\\n")
        out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    if prev is not None:
        cols = list(prev.columns)
        out.append("\\begin{table}[t]\\centering\\small\n\\caption{Prevalencia (\\%) por hallazgo.}\n"
                   "\\label{tab:prevalencia}\n\\begin{tabular}{lccc}\n\\toprule\n"
                   "Hallazgo & " + " & ".join(_tex(c) for c in cols) + " \\\\\n\\midrule\n")
        for name, row in prev.iloc[::-1].iterrows():
            out.append(f"{_tex(name)} & " + " & ".join(f"{v:.1f}" for v in row.values) + " \\\\\n")
        out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return "".join(out)


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_dirs = find_runs(args.runs_dir, args.search_root)
    if not run_dirs:
        sys.exit("No se encontraron experimentos (metrics.json). Añadan como Input P1_Entrenamiento "
                 "y P1_Experimentos2, o ejecuten antes el paso 7.")
    print("Experimentos encontrados:", ", ".join(run_dirs))
    metrics = {r: json.load(open(os.path.join(d, "metrics.json"))) for r, d in run_dirs.items()}

    sub_path = optional_file("subset.csv", "outputs/subset/subset.csv", args.search_root)
    subset = pd.read_csv(sub_path) if sub_path else None
    paired_path = optional_file("paired_comparisons.csv", os.path.join(args.runs_dir, "paired_comparisons.csv"),
                                args.search_root)
    paired = pd.read_csv(paired_path) if paired_path else None
    lang_path = optional_file("validation_per_language.csv", "outputs/labels/validation_per_language.csv",
                              args.search_root)

    print(f"\nFiguras en {args.out_dir}/")
    prev = fig_prevalence(subset, args.out_dir) if subset is not None else None
    fig_macro(metrics, args.out_dir)
    fig_heatmap(metrics, args.out_dir)
    best = args.best if args.best in metrics else max(metrics, key=lambda r: metrics[r]["test_macro_auc"])
    if subset is not None:
        fig_roc(best, run_dirs[best], subset, metrics, args.out_dir)
    if paired is not None and len(paired):
        fig_comparisons(paired, args.out_dir)
    histories = {r: pd.read_csv(os.path.join(d, "history.csv")) for r, d in run_dirs.items()
                 if os.path.exists(os.path.join(d, "history.csv"))}
    if histories:
        fig_history(histories, args.out_dir)
    if lang_path:
        val_lang = pd.read_csv(lang_path, index_col=0)
        if {"recall", "estudios"} <= set(val_lang.columns):
            fig_language(val_lang, args.out_dir)

    tex = tables_tex(metrics, best, paired, prev)
    with open(os.path.join(args.out_dir, "tablas.tex"), "w") as f:
        f.write(tex)
    print("  tablas.tex")
    if prev is not None:
        prev.round(1).to_csv(os.path.join(args.out_dir, "prevalencia.csv"))
    print("\nListo. Estas figuras solo contienen estadísticas agregadas (sin imágenes ni texto de reportes).")


if __name__ == "__main__":
    main()
