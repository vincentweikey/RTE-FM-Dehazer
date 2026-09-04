#!/usr/bin/env python3
import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

from models.unet import EfficientUNet
from flow import FlowMatcher
from tools.build_latent_pair import VaeEncoderDecoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="输入图片文件夹")
    parser.add_argument("-o", "--output", help="输出文件夹（默认=input同级）")
    parser.add_argument("--ckpt", required=True, help="flow 权重路径 *.pt")
    #parser.add_argument("--vae", default="stabilityai/stable-diffusion-xl-refiner-1.0", help="VAE repo or local path")
    parser.add_argument("--vae", default="stabilityai/stable-diffusion-2-1", help="VAE repo or local path")
    parser.add_argument("--steps", type=int, default=50, help="flow 采样步数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = args.output or (args.input.rstrip("/") + "_clean")
    os.makedirs(out_dir, exist_ok=True)

    # 1. VAE & Flow init
    vae = VaeEncoderDecoder(vae_id=args.vae, device=args.device)
    unet = EfficientUNet(
        in_channels=4,
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[8, 16],
        dropout=0.1,
        channel_mult=[1, 2, 4, 4],
        use_scale_shift_norm=True
    ).to(args.device)
    flow = FlowMatcher(unet).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    flow.load_state_dict(ckpt["model"])
    flow.eval()

    # 2. preprocess VAE encode 512
    transform_512 = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x * 2 - 1)          # -> [-1,1]
    ])

    # 3. inference by pathes
    img_paths = [os.path.join(root, f)
                 for root, _, files in os.walk(args.input)
                 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for path in tqdm(img_paths, desc="Inference"):
        rel_path = os.path.relpath(path, args.input)
        save_path = os.path.join(out_dir, os.path.splitext(rel_path)[0] + "_clean.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        original_img = Image.open(path).convert("RGB")
        w_orig, h_orig = original_img.size

        # 512 encode
        img_512 = original_img.resize((512, 512), Image.LANCZOS)
        x = transform_512(img_512).unsqueeze(0).to(args.device)
        latent = vae.encode_image(img_512)          # [1,4,64,64]

        # Flow generate
        clean_latent = flow.generate(latent, num_steps=args.steps)

        # decode resize back
        clean_512 = vae.decode_latent(clean_latent)
        clean_orig = clean_512.resize((w_orig, h_orig), Image.LANCZOS)
        clean_orig.save(save_path)

    print(f"finished -> {out_dir}")


if __name__ == "__main__":
    main()
