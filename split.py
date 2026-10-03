#!/usr/bin/env python3
"""split.py — bagi dataset_raw menjadi train/val dengan menghindari data leakage.

Frame beruntun sangat mirip. Kalau frame tetangga masuk ke train dan val sekaligus,
akurasi validasi terlihat terlalu bagus (slide 22). Ada dua mode:

  sesi : satu sesi utuh (tanggal, kondisi_cahaya, sesi) masuk ke SATU sisi saja.
         Paling ketat. Dipakai kalau tiap kelas punya >= 4 sesi.
  blok : tiap sesi dipotong jadi blok frame berurutan (default 10 frame); blok
         utuh masuk ke satu sisi, dan SETIAP sesi menyumbang blok ke train maupun
         val. Dipakai kalau sesi masih sedikit (< 4 per kelas), supaya kondisi
         cahaya tidak terpisah total antara train dan val.
  auto : pilih sesi jika memungkinkan, jika tidak pakai blok (default).

Pemakaian: python split.py [--mode auto|sesi|blok] [--block 10] [--val-frac 0.2] [--seed 42]
"""
import argparse
import csv
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

MIN_SESI_UNTUK_MODE_SESI = 4


def frame_index(path):
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return 0


def pick_val(units, val_frac, rng):
    """Pilih unit (sesi atau blok) untuk val; selalu sisakan minimal 1 unit untuk train."""
    names = sorted(units)
    rng.shuffle(names)
    total = sum(len(v) for v in units.values())
    target = math.ceil(val_frac * total)
    chosen, n = set(), 0
    for g in names:
        if n < target and len(chosen) < len(names) - 1:
            chosen.add(g)
            n += len(units[g])
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="dataset_raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--mode", choices=["auto", "sesi", "blok"], default="auto")
    ap.add_argument("--block", type=int, default=10, help="jumlah frame berurutan per blok (mode blok)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    meta = raw / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"{meta} tidak ditemukan. Jalankan capture.py dulu.")

    # kelas -> sesi -> daftar (path, kondisi)
    data = defaultdict(lambda: defaultdict(list))
    with open(meta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path = raw / r["kelas"] / r["nama_file"]
            if not path.exists():
                continue
            sesi_key = f'{r["tanggal"]}|{r["kondisi_cahaya"]}|{r.get("sesi") or "1"}'
            data[r["kelas"]][sesi_key].append((path, r["kondisi_cahaya"]))
    if not data:
        raise SystemExit("Tidak ada citra yang cocok dengan metadata.csv.")

    min_sesi = min(len(s) for s in data.values())
    mode = args.mode
    if mode == "auto":
        mode = "sesi" if min_sesi >= MIN_SESI_UNTUK_MODE_SESI else "blok"
    if mode == "sesi" and min_sesi < 2:
        raise SystemExit("Mode sesi butuh >= 2 sesi per kelas. Tambah sesi atau pakai --mode blok.")
    print(f"Mode split: {mode} (sesi paling sedikit per kelas: {min_sesi})")
    if mode == "blok":
        print("  ! Mode blok: val berasal dari sesi/kondisi yang sama dengan train (hanya posisi/waktu berbeda),\n"
              "    jadi akurasi val cenderung OPTIMIS. Tambah sesi baru (hari/lokasi/orang berbeda) untuk uji yang lebih jujur.")

    rng = random.Random(args.seed)
    if out.exists():
        shutil.rmtree(out)

    report, komposisi = [], []
    for kelas in sorted(data):
        sessions = data[kelas]
        files_by_side = {"train": [], "val": []}

        if mode == "sesi":
            units = {k: [p for p, _ in v] for k, v in sessions.items()}
            val_units = pick_val(units, args.val_frac, rng)
            for k, v in sessions.items():
                files_by_side["val" if k in val_units else "train"].extend(v)
        else:
            for k, v in sessions.items():
                blocks = defaultdict(list)
                for item in sorted(v, key=lambda t: frame_index(t[0])):
                    blocks[frame_index(item[0]) // args.block].append(item)
                if len(blocks) < 2:
                    raise SystemExit(f"Sesi {k} (kelas {kelas}) terlalu pendek untuk dipecah jadi blok; "
                                     "tambah citra atau kecilkan --block.")
                val_blocks = pick_val({b: x for b, x in blocks.items()}, args.val_frac, rng)
                for b, x in blocks.items():
                    files_by_side["val" if b in val_blocks else "train"].extend(x)

        for side, items in files_by_side.items():
            dest = out / side / kelas
            dest.mkdir(parents=True, exist_ok=True)
            for p, _ in items:
                shutil.copy2(p, dest / p.name)
            for kond in sorted({c for _, c in items}):
                komposisi.append((kelas, side, kond, sum(1 for _, c in items if c == kond)))
        report.append((kelas, len(files_by_side["train"]), len(files_by_side["val"])))

    print(f"\n{'kelas':<14}{'train':>7}{'val':>6}")
    for kelas, ntr, nva in report:
        print(f"{kelas:<14}{ntr:>7}{nva:>6}")
        if nva < 10:
            print(f"  ! val kelas '{kelas}' hanya {nva} citra, angka akurasi kurang bisa dipercaya.")

    print("\nKomposisi kondisi cahaya (cek: kondisi yang sama harus ada di train DAN val, di semua kelas):")
    print(f"{'kelas':<14}{'sisi':<7}{'kondisi':<10}{'jumlah':>7}")
    for kelas, side, kond, n in komposisi:
        print(f"{kelas:<14}{side:<7}{kond:<10}{n:>7}")
    for kelas in sorted(data):
        tr = {k for c, s, k, _ in komposisi if c == kelas and s == "train"}
        va = {k for c, s, k, _ in komposisi if c == kelas and s == "val"}
        if tr != va:
            print(f"  ! Kelas '{kelas}': kondisi cahaya train {sorted(tr)} != val {sorted(va)} "
                  "(model bisa belajar pintasan dari cahaya).")

    with open(out / "split_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kelas", "sisi", "kondisi", "jumlah"])
        w.writerows(komposisi)
    print(f"\nSelesai -> {out}/train dan {out}/val")


if __name__ == "__main__":
    main()
