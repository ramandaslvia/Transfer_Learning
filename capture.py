#!/usr/bin/env python3
"""capture.py — ambil citra dari kamera robot (opsional di-undistort) + catat metadata.

Contoh:
  python capture.py tangga terang --sesi 1
  python capture.py manusia redup --sesi 2 --lokasi koridor --calib calib.npz
  python capture.py lantai_datar terang --source rekaman.mp4

Tombol: SPASI = simpan frame | Q atau ESC = keluar
Hasil : dataset_raw/<kelas>/<kelas>_<tanggal>_<lokasi>_<kondisi>_<nnn>.png
        dataset_raw/metadata.csv
"""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import cv2
import numpy as np

HEADER = ["nama_file", "kelas", "tanggal", "kondisi_cahaya", "lokasi", "sesi", "catatan"]


def load_calibration(path):
    """Baca matriks kamera K dan koefisien distorsi dari file .npz (hasil pertemuan 2)."""
    d = np.load(path)

    def pick(*names):
        for n in names:
            if n in d.files:
                return d[n]
        raise KeyError(f"Kunci {names} tidak ada di {path}. Kunci tersedia: {d.files}")

    return pick("K", "camera_matrix", "mtx"), pick("dist", "dist_coeffs", "distCoeffs")


class Undistorter:
    """Koreksi distorsi lensa. Resolusi capture harus sama dengan resolusi saat kalibrasi."""

    def __init__(self, K, dist):
        self.K, self.dist, self.maps = K, dist, None

    def __call__(self, frame):
        if self.maps is None:
            h, w = frame.shape[:2]
            new_k, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 0)
            self.maps = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, new_k, (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR)


def next_index(folder, prefix):
    nums = []
    for f in folder.glob(prefix + "*.png"):
        try:
            nums.append(int(f.stem.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    return max(nums, default=0) + 1


def check_token(name, value):
    if not value or any(c.isspace() for c in value):
        sys.exit(f"{name} tidak boleh kosong atau mengandung spasi: '{value}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kelas", help="nama kelas: tangga | manusia | lantai_datar")
    ap.add_argument("kondisi", help="kondisi cahaya: terang | redup | jendela | bayangan ...")
    ap.add_argument("--sesi", type=int, default=1, help="nomor sesi pengambilan (dipakai untuk split anti-leakage)")
    ap.add_argument("--lokasi", default="lab", help="lokasi pengambilan, tanpa spasi (mis. lab, koridor, tangga_gedungA)")
    ap.add_argument("--catatan", default="", help="catatan bebas (mis. 'tangga turun', 'orang membawa tas')")
    ap.add_argument("--source", default="0", help="indeks kamera (0,1,..) atau path file video")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--calib", help="file .npz kalibrasi kamera (K, dist) untuk undistort")
    ap.add_argument("--out", default="dataset_raw")
    args = ap.parse_args()

    for n, v in (("kelas", args.kelas), ("kondisi", args.kondisi), ("lokasi", args.lokasi)):
        check_token(n, v)

    out = Path(args.out)
    folder = out / args.kelas
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    if not meta_path.exists():
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)

    und = Undistorter(*load_calibration(args.calib)) if args.calib else None
    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Kamera / sumber video tidak bisa dibuka.")

    tanggal = dt.date.today().strftime("%Y%m%d")
    prefix = f"{args.kelas}_{tanggal}_{args.lokasi}_{args.kondisi}_"
    idx = next_index(folder, prefix)
    saved = 0
    delay = 1 if is_cam else 30

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if und:
            frame = und(frame)
        view = frame.copy()
        cv2.putText(view, f"{args.kelas}/{args.kondisi} sesi {args.sesi} | tersimpan: {saved} | SPASI=simpan Q=keluar",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("capture", view)
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord("q")):
            break
        if key == 32:
            name = f"{prefix}{idx:03d}.png"
            cv2.imwrite(str(folder / name), frame)   # simpan frame asli tanpa teks overlay
            with open(meta_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([name, args.kelas, tanggal, args.kondisi, args.lokasi, args.sesi, args.catatan])
            idx += 1
            saved += 1
            print("simpan", name)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. {saved} citra baru disimpan di {folder}")


if __name__ == "__main__":
    main()
