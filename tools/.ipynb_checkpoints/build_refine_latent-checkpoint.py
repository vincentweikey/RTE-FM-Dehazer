#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import torch
import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
import torchvision.transforms as T
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import EfficientUNet
from flow import FlowMatcher


class VaeEncoderDecoderMD5:
    def __init__(
        self,
        ckpt_path: str,
        vae_id: str = "stabilityai/stable-diffusion-2-1",
        scaling_factor: float = 0.18215,
        device=None,
    ):
        self.scaling = scaling_factor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # VAE
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").to(self.device)

        # Flow
        unet = EfficientUNet(
            in_channels=4,
            model_channels=128,
            out_channels=4,
            num_res_blocks=2,
            attention_resolutions=[8, 16],
            dropout=0.1,
            channel_mult=[1, 2, 4, 4],
            use_scale_shift_norm=True,
        ).to(self.device)
        self.flow = FlowMatcher(unet).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.flow.load_state_dict(ckpt["model"])
        self.flow.eval()

        # transforms
        self.trans_512 = T.Compose([
            T.Resize((512, 512), Image.LANCZOS),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1),  # -> [-1,1]
        ])
        self.trans_1024 = T.Compose([
            T.Resize((1024, 1024), Image.LANCZOS),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1),
        ])

    @staticmethod
    def _md5(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _collect_files(root: str, ext: Tuple[str, ...]) -> List[str]:
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(ext):
                    files.append(os.path.join(dirpath, f))
        return files

    @torch.no_grad()
    def _process_one(self, img_path: str, save_root: str, steps: int = 50) -> None:
        md5 = self._md5(img_path)
        save_root = Path(save_root)

        sub_dir = self._pick_subdir(save_root, md5)
        if sub_dir is None:
            return

        target512 = sub_dir / f"{md5}.latent512.pt"
        target1024 = sub_dir / f"{md5}.latent1024.pt"

        img = Image.open(img_path).convert("RGB")

        # 1024
        x1024 = self.trans_1024(img).unsqueeze(0).to(self.device)
        latent1024 = self.scaling * self.vae.encode(x1024).latent_dist.sample()
        torch.save(latent1024.cpu(), target1024)

        # 512 
        x512 = self.trans_512(img).unsqueeze(0).to(self.device)
        latent_haze = self.scaling * self.vae.encode(x512).latent_dist.sample()
        clean_latent512 = self.flow.generate(latent_haze, num_steps=steps)
        torch.save(clean_latent512.cpu(), target512)

    def _pick_subdir(self, save_root: Path, md5: str) -> Path:
        for sub in save_root.iterdir():
            if not sub.is_dir():
                continue
            if (sub / f"{md5}.latent512.pt").exists() or (sub / f"{md5}.latent1024.pt").exists():
                return None
        counts = [int(d.name) for d in save_root.iterdir() if d.is_dir()]
        next_id = 0 if not counts else (max(counts) // 10_000 + 1) * 10_000
        sub = save_root / f"{next_id:06d}"
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def process_parent(self, parent_dir: str, skip_existing: bool = True, steps: int = 50) -> None:
        raw_in = os.path.join(parent_dir, "RAW")
        latent_out = os.path.join(parent_dir, "RefineLatent")  # 512+1024 同目录
        os.makedirs(latent_out, exist_ok=True)

        img_list = self._collect_files(raw_in, (".jpg", ".jpeg", ".png"))
        for img_path in tqdm(img_list, desc="Process"):
            if skip_existing:
                # 提前算 md5 可快速跳过，但仍会在 _pick_subdir 里二次检查
                md5 = self._md5(img_path)
                if self._already_exists(latent_out, md5):
                    continue
            self._process_one(img_path, latent_out, steps)

    # ------------ 二次检查是否存在 ------------
    @staticmethod
    def _already_exists(latent_root: str, md5: str) -> bool:
        for sub in Path(latent_root).iterdir():
            if not sub.is_dir():
                continue
            if (sub / f"{md5}.latent512.pt").exists() or (sub / f"{md5}.latent1024.pt").exists():
                return True
        return False



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parent", required=True, help="父文件夹，其下需含 RAW/ 子目录")
    parser.add_argument("--skip", type=int, default=0, help="1=跳过已存在 md5，0=强制覆盖")
    parser.add_argument("--steps", type=int, default=10, help="flow 采样步数")
    args = parser.parse_args()

    tool = VaeEncoderDecoderMD5(ckpt_path="/root/RTE_FM/ckpts/epoch199.pt")  
    tool.process_parent(args.parent, skip_existing=bool(args.skip), steps=args.steps)
    print("Done！")