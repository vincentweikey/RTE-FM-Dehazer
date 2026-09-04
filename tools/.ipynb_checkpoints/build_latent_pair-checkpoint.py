# build_latent_pair.py
from __future__ import annotations
import os
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
import torchvision.transforms as T
from typing import List, Tuple


class VaeEncoderDecoder:
    def __init__(
        self,
        #vae_id="runwayml/stable-diffusion-v1-5",
        vae_id="stabilityai/stable-diffusion-2-1"
        variant="fp16",
        scaling_factor=0.18215,
        device=None,
    ):
        self.scaling = scaling_factor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").to(self.device)

        # resize process
        self.to_tensor = T.Compose([
            T.Resize((512, 512), Image.BILINEAR),
            T.ToTensor()
        ])

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> torch.Tensor:
        x = self.to_tensor(img).unsqueeze(0) * 2 - 1  # [-1,1]
        latent_dist = self.vae.encode(x.to(self.device)).latent_dist
        return self.scaling * latent_dist.sample()  # [1,4,64,64]


    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        latent tensor (1,4,64,64) -> PIL.Image
        """
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        latent = (1 / self.scaling) * latent.to(self.device)

        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().squeeze(0)
        return T.ToPILImage()(image)

   
    def process_parent(self, parent_dir: str, skip_existing: bool = True) -> None:
        """
        parent_dir must exists RAW/  
        output：
            <parent_dir>/RAW_512/   512×512 图像
            <parent_dir>/Latent/    对应 latent
        """
        raw_in  = os.path.join(parent_dir, "RAW")
        raw_out = os.path.join(parent_dir, "RAW_512")
        lat_out = os.path.join(parent_dir, "Latent")
        os.makedirs(raw_out, exist_ok=True)
        os.makedirs(lat_out, exist_ok=True)

        img_list = self._collect_files(raw_in, (".jpg", ".jpeg", ".png"))
        for img_path in tqdm(img_list, desc="Process"):
            rel_path = os.path.relpath(img_path, raw_in)
            img_save = os.path.join(raw_out, rel_path)
            lat_save = os.path.join(lat_out, os.path.splitext(rel_path)[0] + ".pt")

            # skip for adding data
            if skip_existing and os.path.exists(lat_save):
                continue

            os.makedirs(os.path.dirname(img_save), exist_ok=True)
            os.makedirs(os.path.dirname(lat_save), exist_ok=True)

            try:
                img = Image.open(img_path).convert("RGB")
                img_512 = img.resize((512, 512), Image.BILINEAR)
                latent = self.encode_image(img_512)

                img_512.save(img_save)         
                torch.save(latent.cpu(), lat_save)  
            except Exception as e:
                print(f"[WARN] skip {img_path} : {e}")

    # ---------------- 工具 ----------------
    @staticmethod
    def _collect_files(root: str, ext: Tuple[str, ...]) -> List[str]:
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(ext):
                    files.append(os.path.join(dirpath, f))
        return files


# ----------------------------------------------------------------------
# python build_latent_pair.py -p /data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parent", required=True,
                        help="父文件夹，其下需含 RAW/ 子目录")
    parser.add_argument("--skip", type=int, default=0,
                        help="1=跳过已存在 latent，0=强制覆盖")
    args = parser.parse_args()

    tool = VaeEncoderDecoder()
    tool.process_parent(args.parent, skip_existing=bool(args.skip))
    print("✅ 全部处理完成！")