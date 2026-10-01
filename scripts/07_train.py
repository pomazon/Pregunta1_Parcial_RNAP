"""Paso 7 — Entrenamiento y evaluación de un modelo (un experimento por ejecución).

Ejemplos:
    python scripts/07_train.py --model scratch  --run_name A_scratch_bce_soft
    python scripts/07_train.py --model resnet50 --run_name B_resnet50_bce_soft
    python scripts/07_train.py --model resnet50 --loss focal  --run_name B_resnet50_focal_soft
    python scripts/07_train.py --model resnet50 --labels hard --run_name B_resnet50_bce_hard

Datos   : outputs/subset/subset.csv + outputs/cache/<plane>.npy (pasos 5 y 6, o la salida
          de otro notebook añadida como Input).
Etiquetas:
  train : débiles; "soft" (pos=1, unc=0.5, neg/nm=0) o "hard" (unc=0).
  val   : débiles duras (pos=1). Selección del mejor epoch y de los umbrales.
  test  : las 58 etiquetas gold. Solo se usa al final, una vez.
Salida  : outputs/runs/<run_name>/ config.json, history.csv, metrics.json,
          preds_val.csv, preds_test.csv, best.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.common import find_file  # noqa: E402
from knee.data import KneeStudyDataset, worker_init_fn  # noqa: E402
from knee.labels import LABELS  # noqa: E402
from knee.losses import build_loss  # noqa: E402
from knee.metrics import auc_per_label, best_f1_thresholds, bootstrap_auc, prf_at  # noqa: E402
from knee.models import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento multietiqueta")
    p.add_argument("--subset", default="outputs/subset/subset.csv")
    p.add_argument("--cache_dir", default="outputs/cache")
    p.add_argument("--plane", default="Sagittal", choices=["Sagittal", "Coronal"])
    p.add_argument("--model", default="resnet50", help="scratch o un nombre de timm (resnet50, resnet18, ...)")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--pool", default="attn", choices=["attn", "max", "mean"])
    p.add_argument("--loss", default="bce", choices=["bce", "focal"])
    p.add_argument("--labels", default="soft", choices=["soft", "hard"])
    p.add_argument("--pos_weight", default="sqrt", choices=["none", "sqrt", "full"])
    p.add_argument("--n_pos", type=int, default=16, help="Posiciones 2.5D por estudio")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--bs", type=int, default=0, help="0 = automático (8 scratch, 4 preentrenado)")
    p.add_argument("--lr", type=float, default=0, help="0 = automático (1e-3 scratch, 2e-4 preentrenado)")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta", action="store_true", help="Promediar con el volumen en orden inverso")
    p.add_argument("--run_name", default="")
    p.add_argument("--out_dir", default="outputs/runs")
    p.add_argument("--limit", type=int, default=0, help="Usar solo N estudios por partición (prueba)")
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(torch.flip(x, dims=[1])).float())) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    args = parse_args()
    set_seed(args.seed)
    pretrained = args.model != "scratch" and not args.no_pretrained
    bs = args.bs or (8 if args.model == "scratch" else 4)
    lr = args.lr or (1e-3 if args.model == "scratch" else 2e-4)
    run = args.run_name or f"{args.model}_{args.loss}_{args.labels}_{args.plane[:3].lower()}"
    out = os.path.join(args.out_dir, run)
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experimento: {run} | dispositivo: {device} | bs={bs} lr={lr}")

    # ---------------- datos
    subset = pd.read_csv(find_file("subset.csv", args.subset))
    npy = find_file(f"{args.plane}.npy", os.path.join(args.cache_dir, f"{args.plane}.npy"))
    index = pd.read_csv(os.path.join(os.path.dirname(npy), f"{args.plane}_index.csv"))
    vols = np.load(npy, mmap_mode="r")
    df = subset.merge(index.loc[index.ok, ["StudyInstanceUID", "row"]], on="StudyInstanceUID")
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    print("Estudios:", {k: len(v) for k, v in parts.items()})

    soft_cols = [f"{l}__soft" for l in LABELS] if args.labels == "soft" else LABELS
    y_train = parts["train"][soft_cols].to_numpy(np.float32)
    y_val = parts["val"][LABELS].to_numpy(np.float32)
    y_test = parts["test"][[f"{l}__gold" for l in LABELS]].to_numpy(np.float32)

    prev = np.clip(y_train.mean(0), 1e-3, 1 - 1e-3)
    pw = None if args.pos_weight == "none" else (1 - prev) / prev
    if args.pos_weight == "sqrt":
        pw = np.sqrt(pw)
    if pw is not None:
        print("pos_weight:", dict(zip(LABELS, np.round(pw, 2))))

    mk = lambda part, y, train: DataLoader(  # noqa: E731
        KneeStudyDataset(vols, parts[part].row.values, y, args.n_pos, train=train),
        batch_size=bs, shuffle=train, drop_last=train, num_workers=args.workers,
        pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
        persistent_workers=args.workers > 0)
    dl_train, dl_val, dl_test = mk("train", y_train, True), mk("val", y_val, False), mk("test", y_test, False)

    # ---------------- modelo
    model = build_model(args.model, len(LABELS), args.pool, pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params / 1e6:.1f} M | preentrenado: {pretrained} | pooling: {args.pool}")
    crit = build_loss(args.loss, torch.tensor(pw, dtype=torch.float32) if pw is not None else None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    steps = args.epochs * len(dl_train)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "bs": bs, "lr": lr, "pretrained": pretrained, "run": run,
                   "n_params": n_params, "n": {k: len(v) for k, v in parts.items()}}, f, indent=2)

    # ---------------- entrenamiento
    best, best_ep, history, bad = -1.0, -1, [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = crit(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        p_val = predict(model, dl_val, device)
        aucs = auc_per_label(y_val, p_val)
        macro = float(np.nanmean(aucs))
        history.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_macro_auc": macro,
                        **{f"val_auc_{l}": a for l, a in zip(LABELS, aucs)}, "sec": time.time() - t0})
        flag = ""
        if macro > best:
            best, best_ep, bad, flag = macro, ep, 0, " *"
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
        print(f"ep {ep:02d} | loss {tot / max(n, 1):.4f} | val macro AUC {macro:.4f} | "
              f"{time.time() - t0:.0f}s{flag}", flush=True)
        pd.DataFrame(history).to_csv(os.path.join(out, "history.csv"), index=False)
        if bad >= args.patience:
            print(f"Parada temprana: {args.patience} épocas sin mejorar")
            break

    # ---------------- evaluación final con el mejor epoch
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    p_val = predict(model, dl_val, device, args.tta)
    p_test = predict(model, dl_test, device, args.tta)
    th = best_f1_thresholds(y_val, p_val)

    auc_val = auc_per_label(y_val, p_val)
    auc_test = auc_per_label(y_test, p_test)
    lo, hi, (mlo, mhi) = bootstrap_auc(y_test, p_test)
    prf_t = prf_at(y_test, p_test, th)
    prf_05 = prf_at(y_test, p_test, 0.5)

    table = pd.DataFrame({
        "pos_test": y_test.sum(0).astype(int), "auc_val": auc_val.round(3), "auc_test": auc_test.round(3),
        "ic95_lo": lo.round(3), "ic95_hi": hi.round(3), "umbral_val": th.round(2),
        "precision": prf_t[:, 0].round(3), "recall": prf_t[:, 1].round(3), "f1": prf_t[:, 2].round(3),
        "f1@0.5": prf_05[:, 2].round(3)}, index=LABELS)
    macro_test = float(np.nanmean(auc_test))
    pd.set_option("display.width", 200)
    print(f"\n=== {run} | mejor epoch {best_ep} | val macro AUC {np.nanmean(auc_val):.4f} ===")
    print(f"TEST (58 gold) macro AUC: {macro_test:.4f}  [IC95 {mlo:.3f}–{mhi:.3f}]  | "
          f"macro F1 (umbral de val): {prf_t[:, 2].mean():.3f}")
    print(table.to_string())

    metrics = {"run": run, "best_epoch": best_ep, "val_macro_auc": float(np.nanmean(auc_val)),
               "test_macro_auc": macro_test, "test_macro_auc_ci95": [float(mlo), float(mhi)],
               "test_macro_f1": float(prf_t[:, 2].mean()), "test_macro_precision": float(prf_t[:, 0].mean()),
               "test_macro_recall": float(prf_t[:, 1].mean()), "per_label": table.reset_index()
               .rename(columns={"index": "label"}).to_dict(orient="records"),
               "config": {k: getattr(args, k) for k in ("model", "pool", "loss", "labels", "pos_weight", "plane")}}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    for name, part, pr in (("val", parts["val"], p_val), ("test", parts["test"], p_test)):
        pd.concat([part[["StudyInstanceUID"]], pd.DataFrame(pr, columns=LABELS)], axis=1) \
          .to_csv(os.path.join(out, f"preds_{name}.csv"), index=False)
    print(f"Resultados en {out}/")


if __name__ == "__main__":
    main()
