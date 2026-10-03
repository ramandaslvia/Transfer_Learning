#!/usr/bin/env python3
"""infer.py — contoh inferensi satu citra dengan urutan preprocessing yang SAMA seperti training.

  python infer.py results/feature/best.pt contoh.png
  python infer.py results/feature/best.pt contoh.png --calib calib.npz
Urutan: baca (OpenCV, BGR) -> undistort -> BGR->RGB -> Resize 224 -> Normalisasi ImageNet
"""
import argparse

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from capture import Undistorter, load_calibration
from common import val_tf


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["classes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("image")
    ap.add_argument("--calib")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.ckpt, device)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Tidak bisa membaca {args.image}")
    if args.calib:
        bgr = Undistorter(*load_calibration(args.calib))(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # OpenCV BGR -> RGB (jangan lupa!)
    x = val_tf(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.inference_mode():
        prob = torch.softmax(model(x), dim=1)[0]
    for c, p in sorted(zip(classes, prob.tolist()), key=lambda t: -t[1]):
        print(f"{c:<14}{p*100:6.1f}%")


if __name__ == "__main__":
    main()
