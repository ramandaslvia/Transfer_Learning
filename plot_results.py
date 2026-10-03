#!/usr/bin/env python3
"""plot_results.py — gabungkan hasil 3 mode menjadi tabel (markdown) dan grafik akurasi per epoch.

Jalankan setelah train.py feature / partial / scratch selesai:
  python plot_results.py
Keluaran: results/tabel_hasil.md dan results/akurasi_per_epoch.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["feature", "partial", "scratch"]
YANG_DILATIH = {"feature": "fc saja", "partial": "layer4 + fc", "scratch": "semua"}
root = Path("results")

rows, fig_data = [], {}
for m in MODES:
    s_path, h_path = root / m / "summary.json", root / m / "history.csv"
    if not (s_path.exists() and h_path.exists()):
        print(f"(lewati {m}: belum ada hasil)")
        continue
    s = json.load(open(s_path, encoding="utf-8"))
    with open(h_path) as f:
        h = list(csv.DictReader(f))
    fig_data[m] = h
    e90 = s["epoch_pertama_acc_ge_90"]
    rows.append(
        f"| {m} | {s['bobot_awal']} | {YANG_DILATIH[m]} ({s['parameter_dilatih']:,}) | "
        f"{s['akurasi_val_terbaik']*100:.1f}% | {s['epoch_terbaik']} | "
        f"{e90 if e90 else 'tidak tercapai'} | {s['waktu_latih_detik']} |")

if not rows:
    raise SystemExit("Belum ada hasil di results/. Jalankan train.py dulu.")

header = ("| Mode | Bobot awal | Yang dilatih (jumlah param) | Akurasi val terbaik | Epoch terbaik | "
          "Epoch pertama ≥ 90% | Waktu latih (detik) |\n|---|---|---|---|---|---|---|\n")
(root / "tabel_hasil.md").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

plt.figure(figsize=(7, 4.2))
colors = {"feature": "tab:blue", "partial": "tab:orange", "scratch": "tab:green"}
for m, h in fig_data.items():
    ep = [int(r["epoch"]) for r in h]
    plt.plot(ep, [float(r["val_acc"]) for r in h], "-o", color=colors[m], label=f"{m} (val)")
    plt.plot(ep, [float(r["train_acc"]) for r in h], "--", color=colors[m], alpha=0.5, label=f"{m} (train)")
plt.axhline(0.9, color="gray", lw=0.8, ls=":")
plt.xlabel("Epoch"); plt.ylabel("Akurasi"); plt.ylim(0, 1.02); plt.grid(alpha=0.3)
plt.title("Akurasi per epoch — tangga / manusia / lantai_datar")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(root / "akurasi_per_epoch.png", dpi=150)
print("Tersimpan: results/tabel_hasil.md dan results/akurasi_per_epoch.png")
