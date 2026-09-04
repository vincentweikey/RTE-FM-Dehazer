#!/usr/bin/env python3
"""
latent_interpolate.py
两个 latent 文件线性插值生成过渡图
"""
import os
import torch
import argparse
from build_latent_pair import VaeEncoderDecoder  # 与脚本同目录即可


def parse_args():
    parser = argparse.ArgumentParser(description="Latent 空间插值生成过渡图")
    parser.add_argument("latent_a", help="第一个 latent 文件 .pt")
    parser.add_argument("latent_b", help="第二个 latent 文件 .pt")
    parser.add_argument(
        "--out-dir", default=".", help="输出目录（默认当前目录）"
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="α 步长（默认 0.1）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载两个潜在向量
    latent_a = torch.load(args.latent_a, map_location="cpu")
    latent_b = torch.load(args.latent_b, map_location="cpu")
    if latent_a.dim() == 3:
        latent_a = latent_a.unsqueeze(0)
    if latent_b.dim() == 3:
        latent_b = latent_b.unsqueeze(0)

    # 输出文件名模板：A_name_alpha{x.x}.png
    base_name = os.path.splitext(os.path.basename(args.latent_a))[0]

    # 初始化 VAE
    vae = VaeEncoderDecoder()

    # 插值 & 解码
    alpha = 0.0
    while alpha <= 1.0001:
        interp = (1 - alpha) * latent_a + alpha * latent_b
        img = vae.decode_latent(interp)
        out_path = os.path.join(args.out_dir, f"{base_name}_alpha{alpha:.1f}.png")
        img.save(out_path)
        print(f"Saved → {out_path}")
        alpha += args.alpha_step


if __name__ == "__main__":
    # python latent_interpolate.py A.pt B.pt --alpha-step 0.05 --out-dir outs
    main()