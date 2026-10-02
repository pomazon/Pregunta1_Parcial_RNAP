"""Paso 9 — Interpretabilidad: Grad-CAM + atención entre cortes (E5).

Uso:
    python scripts/09_gradcam.py --run C_resnet50_focal_soft

Para cada hallazgo elegido toma, entre los estudios gold POSITIVOS, los de mayor
probabilidad predicha (verdaderos positivos) y genera:
  * Grad-CAM (Selvaraju et al., 2017), implementado aquí: pesos = media espacial del
    gradiente del logit del hallazgo respecto al mapa de la última capa convolucional;
    mapa = ReLU(sum_k peso_k * mapa_k), reescalado a 256x256.
  * Los 3 cortes con mayor peso de atención para ese hallazgo, con el mapa superpuesto.
  * El perfil de atención a lo largo de las 16 posiciones del volumen.
Además calcula, sobre los 58 gold, DÓNDE mira cada hallazgo a lo largo del eje medial-
lateral: distancia media (ponderada por atención) al centro del volumen, en [0, 1].
Esperado si es anatómicamente plausible: LCA cerca del centro (escotadura intercondílea),
meniscos y compartimentos de artrosis hacia los extremos.

Salida: outputs/gradcam/  <hallazgo>_<k>.png, cases.csv, attention_location.csv
Las figuras contienen imágenes de la competencia: no publicarlas fuera del informe sin
revisar las reglas de uso de datos.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import find_file  # noqa: E402
from knee.data import KneeStudyDataset  # noqa: E402
from knee.labels import LABELS  # noqa: E402
from knee.models import build_model  # noqa: E402

ES = {"ACL": "LCA", "MCL": "LCM", "Medial Meniscus": "Menisco medial", "Lateral Meniscus": "Menisco lateral",
      "Medial OA": "Artrosis medial", "Lateral OA": "Artrosis lateral", "PF OA": "Artrosis patelofemoral",
      "Effusion": "Derrame", "Synovitis": "Sinovitis", "Baker's": "Quiste de Baker",
      "Contusion": "Contusión ósea", "Fracture": "Fractura"}


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM y atención")
    p.add_argument("--run", default="C_resnet50_focal_soft")
    p.add_argument("--runs_dir", default="outputs/runs")
    p.add_argument("--subset", default="outputs/subset/subset.csv")
    p.add_argument("--cache_dir", default="outputs/cache")
    p.add_argument("--labels", default="ACL,Lateral Meniscus,Medial Meniscus,Effusion,Fracture,Baker's")
    p.add_argument("--n_cases", type=int, default=2)
    p.add_argument("--out_dir", default="outputs/gradcam")
    return p.parse_args()


def find_run_dir(run, runs_dir):
    local = os.path.join(runs_dir, run)
    if os.path.isfile(os.path.join(local, "best.pt")):
        return local
    hits = glob.glob(f"/kaggle/input/**/runs/{run}/best.pt", recursive=True)
    if not hits:
        raise FileNotFoundError(f"No se encontró {run}/best.pt (añade como Input el notebook de entrenamiento)")
    print(f"AVISO: se usa {hits[0]}")
    return os.path.dirname(hits[0])


def gradcam_study(model, x, label_idx):
    """x [1, P, 3, H, W] -> cam [P, H, W] en [0,1], atención [P], probabilidad."""
    model.zero_grad(set_to_none=True)
    b, p = x.shape[:2]
    xn = (x.flatten(0, 1) - model.mean) / model.std
    fmap = model.encoder(xn)                                   # [P, C, h, w]
    fmap.retain_grad()
    vec = F.adaptive_avg_pool2d(fmap, 1).flatten(1).view(b, p, -1)
    logits, att = model.aggregate(vec)
    logits[0, label_idx].backward()
    w = fmap.grad.mean(dim=(2, 3), keepdim=True)               # [P, C, 1, 1]
    cam = F.relu((w * fmap).sum(1, keepdim=True))              # [P, 1, h, w]
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1).values.view(-1, 1, 1) + 1e-8)
    a = att[0, :, label_idx] if att is not None else torch.full((p,), 1.0 / p)
    return cam.detach().cpu().numpy(), a.detach().cpu().numpy(), torch.sigmoid(logits[0, label_idx]).item()


def plot_case(img, cam, att, label, uid_short, prob, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = np.argsort(att)[::-1][:3]
    fig = plt.figure(figsize=(13, 3.6))
    for k, s in enumerate(top):
        ax = fig.add_subplot(1, 4, k + 1)
        ax.imshow(img[s], cmap="gray")
        ax.imshow(cam[s], cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"posición {s} · atención {att[s]:.2f}", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(1, 4, 4)
    ax.bar(np.arange(len(att)), att, color="#4C72B0")
    ax.set_xlabel("posición a lo largo del volumen (extremo → extremo)", fontsize=8)
    ax.set_ylabel("peso de atención", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.suptitle(f"{ES.get(label, label)} · estudio {uid_short} · p = {prob:.2f} (gold positivo)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = find_run_dir(args.run, args.runs_dir)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    plane = cfg.get("plane", "Sagittal")
    model = build_model(cfg["model"], len(LABELS), cfg.get("pool", "attn"), pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(True)

    subset = pd.read_csv(find_file("subset.csv", args.subset))
    npy = find_file(f"{plane}.npy", os.path.join(args.cache_dir, f"{plane}.npy"))
    index = pd.read_csv(os.path.join(os.path.dirname(npy), f"{plane}_index.csv"))
    vols = np.load(npy, mmap_mode="r")
    test = subset[subset.split == "test"].merge(index.loc[index.ok, ["StudyInstanceUID", "row"]],
                                                on="StudyInstanceUID").reset_index(drop=True)
    preds = pd.read_csv(os.path.join(run_dir, "preds_test.csv"))
    test = test.merge(preds.rename(columns={l: f"{l}__p" for l in LABELS}), on="StudyInstanceUID")
    ds = KneeStudyDataset(vols, test.row.values, np.zeros((len(test), len(LABELS))), cfg.get("n_pos", 16))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Modelo: {args.run} ({cfg['model']}, plano {plane}) | estudios gold: {len(test)}")

    # ---- dónde mira cada hallazgo (todos los gold, sin gradientes)
    rows = []
    with torch.no_grad():
        for i in range(len(test)):
            x = ds[i][0].unsqueeze(0).to(device)
            _, vec = model.slice_features(x)
            _, att = model.aggregate(vec)
            if att is None:
                break
            a = att[0].cpu().numpy()                                  # [P, L]
            pos = np.linspace(-1, 1, a.shape[0])
            for j, lab in enumerate(LABELS):
                rows.append({"label": lab, "dist_centro": float((a[:, j] * np.abs(pos)).sum()),
                             "gold": test.loc[i, f"{lab}__gold"]})
    if rows:
        loc = pd.DataFrame(rows)
        tab = loc.groupby("label").agg(dist_centro_media=("dist_centro", "mean"),
                                       dist_centro_positivos=("dist_centro", lambda s: s[loc.loc[s.index, "gold"] == 1].mean()))
        tab = tab.reindex(LABELS).round(3)
        print("\n=== ¿Dónde mira cada hallazgo? Distancia media al centro del volumen (0 = centro, 1 = extremo) ===")
        uniform = float(np.abs(np.linspace(-1, 1, cfg.get("n_pos", 16))).mean())
        print(f"Atención uniforme (sin preferencia) = {uniform:.3f}. Esperado: LCA por debajo; "
              "meniscos y artrosis por compartimento por encima.")
        print(tab.to_string())
        tab.to_csv(os.path.join(args.out_dir, "attention_location.csv"))

    # ---- Grad-CAM en verdaderos positivos de mayor confianza
    cases = []
    for lab in [l.strip() for l in args.labels.split(",") if l.strip()]:
        j = LABELS.index(lab)
        pos = test[test[f"{lab}__gold"] == 1].sort_values(f"{lab}__p", ascending=False).head(args.n_cases)
        for k, (i, r) in enumerate(pos.iterrows()):
            x, _ = ds[i]
            cam, att, prob = gradcam_study(model, x.unsqueeze(0).to(device), j)
            img = x[:, 1].numpy()                                     # canal central = corte propio
            path = os.path.join(args.out_dir, f"{lab.replace(' ', '_').replace(chr(39), '')}_{k + 1}.png")
            plot_case(img, cam, att, lab, r.StudyInstanceUID[-6:], prob, path)
            top = [int(t) for t in np.argsort(att)[::-1][:3]]
            cases.append({"label": lab, "case": k + 1, "study_suffix": r.StudyInstanceUID[-6:], "prob": round(prob, 3),
                          "top_positions": ",".join(map(str, top)), "png": os.path.basename(path)})
            print(f"{lab:17s} caso {k + 1}: p={prob:.2f} | posiciones con más atención: {list(top)} -> {path}")
    pd.DataFrame(cases).to_csv(os.path.join(args.out_dir, "cases.csv"), index=False)
    print(f"\nFiguras en {args.out_dir}/")


if __name__ == "__main__":
    main()
