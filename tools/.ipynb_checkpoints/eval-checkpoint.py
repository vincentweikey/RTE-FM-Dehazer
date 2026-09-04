#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_256.py
统一 resize 到 256×256 后，纯 NumPy 计算 PSNR / SSIM (Y channel)
依赖: numpy, opencv-python, tqdm
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


def resize256(img: np.ndarray) -> np.ndarray:
    """cv2 插值双线性 resize 到 256×256"""
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)


def to_y(img: np.ndarray) -> np.ndarray:
    """BGR -> Y (ITU-R BT.601)"""
    if len(img.shape) == 2:
        return img
    return (img * np.array([0.114, 0.587, 0.299]).reshape(1, 1, 3)).sum(axis=2)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """输入 float32，返回 dB"""
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray,
         win=11, sigma=1.5, C1=0.01**2*255**2, C2=0.03**2*255**2) -> float:
    """单通道 SSIM（高斯窗口）"""
    gauss = cv2.getGaussianKernel(win, sigma)
    window = (gauss @ gauss.T).astype(np.float32)
    r = win // 2
    mu1 = cv2.filter2D(img1, -1, window)[r:-r, r:-r]
    mu2 = cv2.filter2D(img2, -1, window)[r:-r, r:-r]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sig1 = cv2.filter2D(img1**2, -1, window)[r:-r, r:-r] - mu1_sq
    sig2 = cv2.filter2D(img2**2, -1, window)[r:-r, r:-r] - mu2_sq
    sig12 = cv2.filter2D(img1*img2, -1, window)[r:-r, r:-r] - mu1_mu2
    num = (2*mu1_mu2 + C1)*(2*sig12 + C2)
    den = (mu1_sq + mu2_sq + C1)*(sig1 + sig2 + C2)
    return float((num / den).mean())


def eval_pair(gt_path: Path, de_path: Path):
    """resize -> Y -> PSNR/SSIM"""
    gt = cv2.imread(str(gt_path))
    de = cv2.imread(str(de_path))
    if gt is None or de is None:
        raise RuntimeError(f"imread failed: {gt_path} or {de_path}")
    # 统一 resize
    gt = resize256(gt)
    de = resize256(de)
    # 转 Y
    gt_y = to_y(gt).astype(np.float32)
    de_y = to_y(de).astype(np.float32)
    return {"PSNR": psnr(gt_y, de_y), "SSIM": ssim(gt_y, de_y)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, type=str)
    parser.add_argument("--dehaze_dir", required=True, type=str)
    parser.add_argument("--output", default="results_256.csv", type=str)
    parser.add_argument("--img_suffix", default=".jpg", type=str)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    de_dir = Path(args.dehaze_dir).resolve()
    assert gt_dir.is_dir() and de_dir.is_dir()

    gt_imgs = sorted(list(gt_dir.glob(f"*{args.img_suffix}")))
    de_map = {p.name: p for p in de_dir.glob(f"*{args.img_suffix}")}

    rows = []
    for gtp in tqdm(gt_imgs, desc="Eval@256"):
        name = gtp.name
        dep = de_map.get(name)
        if dep is None:
            print(f"[WARN] skip {name}")
            continue
        try:
            rows.append({"img": name, **eval_pair(gtp, dep)})
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if not rows:
        print("No pairs evaluated!")
        return

    # 写 CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img", "PSNR", "SSIM"])
        writer.writeheader()
        writer.writerows(rows)

    # 打印平均
    df = pd.DataFrame(rows)
    print("\n===== Average @256 =====")
    print(df.mean(numeric_only=True))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()