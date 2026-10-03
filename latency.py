#!/usr/bin/env python3
"""latency.py — ukur latensi inferensi ResNet-18 vs MobileNetV3-Small (slide 16, langkah 4).

Jalankan DI PERANGKAT TARGET (mis. Raspberry Pi di robot), bukan di laptop,
kalau ingin angka yang relevan. Bobot acak dipakai karena latensi tidak
bergantung pada nilai bobot, hanya pada arsitektur dan ukuran input.

  python latency.py --classes 3 --runs 200 --threads 4
"""
import argparse
import csv
import statistics
import time

import torch
from torchvision import models

BUDGET_MS = 35.0   # jatah inferensi dari anggaran latensi 15 FPS (slide 16)


def bench(name, model, device, runs, warmup, size):
    model = model.eval().to(device)
    x = torch.randn(1, 3, size, size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    mean = statistics.mean(times)
    p95 = times[int(0.95 * len(times)) - 1]
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return [name, f"{params:.2f}", f"{mean:.1f}", f"{statistics.median(times):.1f}", f"{p95:.1f}",
            f"{1000 / mean:.1f}", "ya" if p95 <= BUDGET_MS else "TIDAK"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=0, help="jumlah thread CPU (0 = default torch)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="results/latency_results.csv")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"Device: {device} | threads: {torch.get_num_threads()} | input 1x3x{args.size}x{args.size}")

    cands = [
        ("ResNet-18", models.resnet18(weights=None, num_classes=args.classes)),
        ("MobileNetV3-Small", models.mobilenet_v3_small(weights=None, num_classes=args.classes)),
    ]
    header = ["model", "param_juta", "mean_ms", "median_ms", "p95_ms", "fps_dari_mean", f"p95<=({BUDGET_MS:.0f}ms)"]
    rows = [bench(n, m, device, args.runs, args.warmup, args.size) for n, m in cands]

    print("\n" + " | ".join(header))
    for r in rows:
        print(" | ".join(r))
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nTersimpan: {args.out}")
    print("Catatan: ini hanya waktu forward model. Anggaran penuh juga mencakup akuisisi, preprocess, postprocess, ROS2.")


if __name__ == "__main__":
    main()
