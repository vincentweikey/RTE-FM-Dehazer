# PairedLatentDataset
# Load pair-wise hazy/clean data for training / validation
import os
import torch
import random
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class PairedLatentDataset(Dataset):
    def __init__(self,
                 raw_root: str,
                 latent_root: str,
                 img_size: int = 256,
                 split: str = "train",
                 train_ratio: float = 0.9,
                 seed: int = 42):
        self.raw_root    = raw_root
        self.latent_root = latent_root
        self.split       = split
        self.transform   = T.Compose([
            T.Resize((img_size, img_size), Image.BILINEAR),
            T.ToTensor(),                      # 0~1
            T.Lambda(lambda x: x * 2 - 1)      # -> [-1,1]
        ])

        # 1. collection all pair-wise data
        all_samples = []
        for root, _, files in os.walk(raw_root):
            for f in files:
                if f.endswith("_hazy_.jpg"):
                    md5 = f.replace("_hazy_.jpg", "")
                    clean_img = os.path.join(root, md5 + ".jpg")
                    hazy_img  = os.path.join(root, f)
                    rel_dir   = os.path.relpath(root, raw_root)
                    clean_pt  = os.path.join(latent_root, rel_dir, md5 + ".pt")
                    hazy_pt   = os.path.join(latent_root, rel_dir, md5 + "_hazy_.pt")
                    if all(map(os.path.exists, [clean_img, hazy_img, clean_pt, hazy_pt])):
                        all_samples.append({
                            "clean_img": clean_img,
                            "hazy_img":  hazy_img,
                            "clean_latent": clean_pt,
                            "hazy_latent":  hazy_pt
                        })
        # 2. fix seed 
        random.seed(seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        print(f"[Dataset {split}] {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "clean_img": self.transform(Image.open(s["clean_img"]).convert("RGB")),
            "hazy_img":  self.transform(Image.open(s["hazy_img"]).convert("RGB")),
            "clean_latent": torch.load(s["clean_latent"]).squeeze(0),
            "hazy_latent":  torch.load(s["hazy_latent"]).squeeze(0)
        }

class PairedRefineDataset(Dataset):
    def __init__(self,
                 latent_root: str,
                 split: str = "train",
                 train_ratio: float = 0.9,
                 seed: int = 42):
        self.latent_root = Path(latent_root)
        self.split = split

        all_samples = []
        for p512 in self.latent_root.rglob("*.latent512.pt"):
            md5 = p512.stem.replace(".latent512", "")
            p1024 = p512.with_name(f"{md5}.latent1024.pt")
            if p1024.exists():
                all_samples.append({"md5": md5,
                                    "latent512": str(p512),
                                    "latent1024": str(p1024)})

        random.seed(seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        print(f"[PairedRefineDataset {split}] {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "latent512":  torch.load(s["latent512"]).squeeze(0),   # [4,64,64]
            "latent1024": torch.load(s["latent1024"]).squeeze(0)   # [4,128,128]
        }





# ================== local run test ==================
if __name__ == "__main__":
    raw   = "/root/autodl-tmp/data/RAW_512/"      
    latent= "/root/autodl-tmp/data/Latent/"

    train_ds = PairedLatentDataset(raw, latent, split="train")
    val_ds   = PairedLatentDataset(raw, latent, split="val")

    print("Train samples:", len(train_ds))
    print("Val   samples:", len(val_ds))

    #for name, ds in zip(["train", "val"], [train_ds, val_ds]):
    #    idx = random.randint(0, len(ds) - 1)
    #    sample = ds[idx]
    #    print(f"\n{name} sample shapes:")
    #    for k, v in sample.items():
    #        print(f"  {k:12}: {v.shape}")

    ds_train = PairedRefineDataset("/root/autodl-tmp/data/RefineLatent", split="train")
    ds_val   = PairedRefineDataset("/root/autodl-tmp/data/RefineLatent", split="val")
    print("train:", len(ds_train), "val:", len(ds_val))
    sample = ds_train[0]
    for k, v in sample.items():
        print(k, v.shape)

