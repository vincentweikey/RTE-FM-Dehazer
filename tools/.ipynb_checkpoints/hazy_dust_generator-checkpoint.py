# add_dust_pixelwise_v2.py
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class PixelAddDust:
    def __init__(
        self,
        root_A: str,
        root_B: str,
        save_root: str,
        p_dust: float = 0.8,
        dust_color_perturb: float = 0.08,
        alpha_max: float = 0.7,  # 最远深度处的最大不透明度
    ):
        self.root_A = Path(root_A)
        self.root_B = Path(root_B)
        self.save_root = Path(save_root)
        self.p_dust = p_dust
        self.dust_color_perturb = dust_color_perturb
        self.alpha_max = alpha_max

        self.dust_list = [
            str(p) for p in self.root_B.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        assert self.dust_list, "未在文件夹B中找到任何灰尘图片！"

    # ---------- 工具 ---------- #
    @staticmethod
    def _imread(p, flag=cv2.IMREAD_COLOR):
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flag)

    @staticmethod
    def _imwrite(p, img):
        cv2.imencode(".jpg", img)[1].tofile(str(p))

    def _random_crop_square(self, img):
        h, w = img.shape[:2]
        short = min(h, w)
        size = int(random.uniform(0.8, 1.0) * short)
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return img[y : y + size, x : x + size]

    # ---------- 深度拉伸 ---------- #
    @staticmethod
    def _normalize_depth(depth):
        d = depth.astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            return 255 * (d - d_min) / (d_max - d_min)
        return np.zeros_like(d)

    # ---------- 主逻辑 ---------- #
    def _add_dust_pixelwise(self, img_bgr, depth):
        h, w = img_bgr.shape[:2]
        # 1. 深度拉伸 0-255
        depth_norm = self._normalize_depth(depth)
        # 2. 随机灰尘图
        dust = self._imread(random.choice(self.dust_list))
        dust = self._random_crop_square(dust)
        dust = cv2.resize(dust, (w, h))
        # 3. 颜色扰动
        noise = np.random.normal(0, self.dust_color_perturb * 255, dust.shape)
        dust = np.clip(dust + noise, 0, 255).astype(np.uint8)
        # 4. 深度控制透明度
        alpha = np.clip(depth_norm / 255.0 * self.alpha_max, 0, self.alpha_max)[..., None]
        # 5. 像素相加
        blended = img_bgr.astype(np.float32) * (1 - alpha) + dust.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _process_one_pair(self, img_path, depth_path):
        img = self._imread(img_path)
        depth = self._imread(depth_path, cv2.IMREAD_GRAYSCALE)

        # 1. 复制原图
        rel_dir = img_path.relative_to(self.root_A).parent
        out_dir = self.save_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / img_path.name).write_bytes(img_path.read_bytes())

        # 2. 概率加灰尘
        if random.random() < self.p_dust:
            img = self._add_dust_pixelwise(img, depth)

        # 3. 保存结果
        self._imwrite(out_dir / (img_path.stem + "_hazy_.jpg"), img)

    def run(self):
        img_list = [
            p for p in self.root_A.rglob("*.jpg")
            if not p.name.endswith("_depth.jpg") and not p.name.endswith("_hazy_.jpg")
        ]
        for img_p in tqdm(img_list, desc="AddDust"):
            depth_p = img_p.with_name(img_p.stem + "_depth.jpg")
            if not depth_p.exists():
                print(f"[WARN] 未找到对应深度图，跳过：{img_p}")
                continue
            self._process_one_pair(img_p, depth_p)


# -------------------- 使用 -------------------- #
if __name__ == "__main__":
    gen = PixelAddDust(
        root_A=r"/root/autodl-tmp/synHazy/RAW",
        root_B=r"/root/autodl-tmp/synHazy/Dust",
        save_root=r"./result",
        p_dust=0.8,
    )
    gen.run()

