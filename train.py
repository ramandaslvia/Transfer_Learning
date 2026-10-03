#!/usr/bin/env python3
"""train.py — bandingkan tiga pendekatan pada ResNet-18 (slide 21).

  feature : bobot ImageNet, hanya fc yang dilatih      (lr 1e-3)
  partial : bobot ImageNet, layer4 + fc dilatih        (lr 1e-4 / 1e-3)
  scratch : bobot acak, semua dilatih                  (lr 1e-3)

Pemakaian:
  python train.py feature
  python train.py partial
  python train.py scratch
  python train.py feature --epochs 10 --batch 16 --data dataset

Keluaran di results/<mode>/: history.csv, summary.json, confusion.csv, best.pt
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from common import train_tf, val_tf


def build_model(mode, num_classes):
    if mode == "scratch":
        m = models.resnet18(weights=None)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)      # head baru: selalu bisa dilatih

    for name, p in m.named_parameters():
        if mode == "feature":
            p.requires_grad = name.startswith("fc.")
        elif mode == "partial":
            p.requires_grad = name.startswith(("layer4.", "fc."))
        else:
            p.requires_grad = True
    return m


def build_optimizer(m, mode):
    if mode == "feature":
        return torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    if mode == "partial":
        return torch.optim.Adam([
            {"params": m.layer4.parameters(), "lr": 1e-4},
            {"params": m.fc.parameters(), "lr": 1e-3},
        ])
    return torch.optim.Adam(m.parameters(), lr=1e-3)


def set_train_mode(m, mode):
    """Lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah (slide 11)."""
    m.train()
    if mode == "feature":
        for name, child in m.named_children():
            if name != "fc":
                child.eval()
    elif mode == "partial":
        for name in ("conv1", "bn1", "layer1", "layer2", "layer3"):
            getattr(m, name).eval()


@torch.no_grad()
def evaluate(model, loader, device, criterion, n_cls):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        for t, p in zip(y.cpu(), pred.cpu()):
            conf[t, p] += 1
    return loss_sum / total, correct / total, conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["feature", "partial", "scratch"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(Path(args.data) / "train", train_tf)
    val_ds = datasets.ImageFolder(Path(args.data) / "val", val_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"Kelas train {train_ds.classes} != kelas val {val_ds.classes}")
    classes = train_ds.classes
    n_cls = len(classes)
    print(f"Mode={args.mode} | device={device} | kelas={classes} | train={len(train_ds)} val={len(val_ds)}")

    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_ld = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_model(args.mode, n_cls).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter yang dilatih: {n_train_params:,}")

    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(model, args.mode)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"acc": -1.0}
    first_90, train_time = None, 0.0

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        set_train_mode(model, args.mode)
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()
        tr_loss, tr_acc = loss_sum / total, correct / total
        va_loss, va_acc, conf = evaluate(model, val_ld, device, criterion, n_cls)
        dt_ep = time.perf_counter() - t0
        train_time += dt_ep

        history.append([ep, tr_loss, tr_acc, va_loss, va_acc, dt_ep])
        print(f"ep {ep:2d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {dt_ep:.1f}s")

        if first_90 is None and va_acc >= 0.90:
            first_90 = ep
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "epoch": ep, "conf": conf.tolist()}
            torch.save({"state_dict": model.state_dict(), "classes": classes, "mode": args.mode},
                       out_dir / "best.pt")

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "detik"])
        w.writerows(history)
    with open(out_dir / "confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benar\\prediksi"] + classes)
        for name, row in zip(classes, best["conf"]):
            w.writerow([name] + row)
    summary = {
        "mode": args.mode,
        "bobot_awal": "acak" if args.mode == "scratch" else "ImageNet",
        "parameter_dilatih": n_train_params,
        "akurasi_val_terbaik": best["acc"],
        "epoch_terbaik": best["epoch"],
        "epoch_pertama_acc_ge_90": first_90,
        "waktu_latih_detik": round(train_time, 1),
        "kelas": classes,
        "device": str(device),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nRingkasan:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
