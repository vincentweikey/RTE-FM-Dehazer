#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register_crop_hazy.py
对文件夹 A 中所有 md5.jpg / md5_hazy_.jpg 做配准、warp、公共域 crop，
结果保持子目录结构输出到 A_crop。
"""
import os
import cv2
import torch
import argparse
import numpy as np
from glob import glob
from os.path import join, dirname, basename, splitext, relpath

from tools import get_padding_size
from networks.dkm.models.model_zoo.DKMv3 import DKMv3
from networks.roma.roma import RoMa
from networks.loftr.loftr import LoFTR
from networks.loftr.misc import lower_config
from networks.loftr.config import get_cfg_defaults
from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue

# ---------- 通用工具 ----------
def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), mode)
    if img is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(img.shape) == 3:
        img = img[:, :, ::-1]  # BGR → RGB
    return img


def preprocess(image: np.ndarray, grayscale=False, resize_max=1024, dfactor=8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        sc = resize_max / max(size)
        if sc < 1.0:
            new_size = tuple(int(round(x * sc)) for x in size)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            scale = np.array(size) / np.array(new_size)

    if grayscale:
        image = image[None]
    else:
        image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image / 255.0).float()

    # 保证尺寸能被 dfactor 整除
    h, w = image.shape[-2:]
    new_h, new_w = h // dfactor * dfactor, w // dfactor * dfactor
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    scale = np.array(size) / np.array([new_w, new_h])
    return image, scale


def build_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = None
    model = None
    detector = None

    if args.model == 'gim_dkm':
        ckpt = 'gim_dkm_100h.ckpt'
        model = DKMv3(weights=None, h=672, w=896)
    elif args.model == 'gim_roma':
        ckpt = 'gim_roma_100h.ckpt'
        model = RoMa(img_size=[672])
    elif args.model == 'gim_loftr':
        ckpt = 'gim_loftr_50h.ckpt'
        model = LoFTR(lower_config(get_cfg_defaults())['loftr'])
    elif args.model == 'gim_lightglue':
        ckpt = 'gim_lightglue_100h.ckpt'
        detector = SuperPoint({
            'max_num_keypoints': 2048,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        })
        model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })
    else:
        raise ValueError(f'Unknown model {args.model}')

    # ---------- load weights ----------
    ckpt_path = join('weights', ckpt)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if args.model == 'gim_dkm':
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif args.model == 'gim_roma':
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif args.model == 'gim_loftr':
        model.load_state_dict(state_dict)
    elif args.model == 'gim_lightglue':
        det_dict = {k.replace('superpoint.', ''): v for k, v in state_dict.items() if k.startswith('superpoint.')}
        detector.load_state_dict(det_dict)
        lg_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(lg_dict)

    if detector is not None:
        detector = detector.eval().to(device)
    model = model.eval().to(device)
    return model, detector, device


def get_pairs(root):
    """返回 [(clean_path, hazy_path), ...]"""
    pairs = []
    for cln in glob(join(root, "**", "*.jpg"), recursive=True):
        if "_hazy_" in cln:
            continue
        base, ext = splitext(cln)
        hzy = base + "_hazy_" + ext
        if os.path.isfile(hzy):
            pairs.append((cln, hzy))
    return pairs


def register_once(model, detector, device, path0, path1, args):
    """返回 warp 后的 hazy 图（uint8 BGR）及有效 mask（0/255）"""
    image0 = read_image(path0)
    image1 = read_image(path1)

    image0, scale0 = preprocess(image0, grayscale=False, resize_max=args.resize_max)
    image1, scale1 = preprocess(image1, grayscale=False, resize_max=args.resize_max)

    image0 = image0.unsqueeze(0).to(device)
    image1 = image1.unsqueeze(0).to(device)

    kpts0, kpts1 = None, None

    # ---------- DKM ----------
    if args.model == 'gim_dkm':
        h_net, w_net = 672, 896
        orig_w0, orig_h0, pad_l0, pad_r0, pad_t0, pad_b0 = get_padding_size(image0, w_net, h_net)
        orig_w1, orig_h1, pad_l1, pad_r1, pad_t1, pad_b1 = get_padding_size(image1, w_net, h_net)
        image0_ = torch.nn.functional.pad(image0, (pad_l0, pad_r0, pad_t0, pad_b0))
        image1_ = torch.nn.functional.pad(image1, (pad_l1, pad_r1, pad_t1, pad_b1))

        with torch.no_grad():
            dense_matches, dense_certainty = model.match(image0_, image1_)
            sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 5000)

        h0, w0 = image0_.shape[-2:]
        h1, w1 = image1_.shape[-2:]
        kpts0 = torch.stack((w0 * (sparse_matches[:, 0] + 1) / 2, h0 * (sparse_matches[:, 1] + 1) / 2), dim=-1)
        kpts1 = torch.stack((w1 * (sparse_matches[:, 2] + 1) / 2, h1 * (sparse_matches[:, 3] + 1) / 2), dim=-1)

        kpts0 -= kpts0.new_tensor([pad_l0, pad_t0])
        kpts1 -= kpts1.new_tensor([pad_l1, pad_t1])
        mask = (kpts0[:, 0] > 0) & (kpts0[:, 1] > 0) & \
               (kpts1[:, 0] > 0) & (kpts1[:, 1] > 0)
        mask = mask & (kpts0[:, 0] <= orig_w0 - 1) & (kpts0[:, 1] <= orig_h0 - 1) & \
                      (kpts1[:, 0] <= orig_w1 - 1) & (kpts1[:, 1] <= orig_h1 - 1)
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        mconf = mconf[mask]

    # ---------- RoMA ----------
    elif args.model == 'gim_roma':
        with torch.no_grad():
            warp01 = model({'image0': image0, 'image1': image1})
        kpts0 = warp01['kpts0']
        kpts1 = warp01['kpts1']
        mconf = warp01['conf']

    # ---------- LoFTR ----------
    elif args.model == 'gim_loftr':
        data = dict(image0=image0, image1=image1)
        with torch.no_grad():
            model(data)
        kpts0 = data['mkpts0_f']
        kpts1 = data['mkpts1_f']
        mconf = data['mconf']

    # ---------- LightGlue ----------
    elif args.model == 'gim_lightglue':
        gray0 = read_image(path0, grayscale=True)
        gray1 = read_image(path1, grayscale=True)
        gray0, _ = preprocess(gray0, grayscale=True, resize_max=args.resize_max)
        gray1, _ = preprocess(gray1, grayscale=True, resize_max=args.resize_max)
        gray0 = gray0.unsqueeze(0).to(device)
        gray1 = gray1.unsqueeze(0).to(device)

        pred = {}
        with torch.no_grad():
            pred.update({k + '0': v for k, v in detector({"image": gray0}).items()})
            pred.update({k + '1': v for k, v in detector({"image": gray1}).items()})
            pred.update(model({**pred, 'image0': gray0, 'image1': gray1,
                               'image_size0': torch.tensor(gray0.shape[-2:][::-1])[None],
                               'image_size1': torch.tensor(gray1.shape[-2:][::-1])[None]}))
        kpts0 = pred['keypoints0'][0] * scale0
        kpts1 = pred['keypoints1'][0] * scale1
        matches = pred['matches'][0]
        mconf = pred['scores'][0]
        kpts0 = kpts0[matches[..., 0]]
        kpts1 = kpts1[matches[..., 1]]

    # ---------- 几何估计 ----------
    if len(kpts0) < 8:
        return None, None
    F, inliers = cv2.findFundamentalMat(kpts0.cpu().numpy(), kpts1.cpu().numpy(),
                                        cv2.USAC_MAGSAC, 1.0, 0.999999, 10000)
    inliers = inliers.ravel() > 0
    if F is None or inliers.sum() < 8:
        return None, None

    # ---------- warp ----------
    H, _ = cv2.findHomography(kpts1.cpu().numpy(), kpts0.cpu().numpy(),
                              cv2.USAC_MAGSAC, 8.0, 0.999, 10000)
    if H is None:
        return None, None

    raw0 = (image0[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[..., ::-1]
    raw1 = (image1[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[..., ::-1]

    warp_hazy = cv2.warpPerspective(raw1, H, (raw0.shape[1], raw0.shape[0]))
    mask = (warp_hazy.sum(axis=2) > 0).astype(np.uint8) * 255
    return warp_hazy, mask


def crop_valid(img, mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img, mask
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    return img[y1:y2 + 1, x1:x2 + 1], mask[y1:y2 + 1, x1:x2 + 1]

from tqdm import tqdm
import traceback   # 可选：打印详细异常栈
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_A', required=True, help="原始数据根目录")
    parser.add_argument('--model', default='gim_dkm', choices=['gim_dkm', 'gim_roma', 'gim_loftr', 'gim_lightglue'])
    parser.add_argument('--resize_max', type=int, default=1024)
    args = parser.parse_args()

    model, detector, device = build_model(args)
    pairs = get_pairs(args.root_A)
    print(f'共找到 {len(pairs)} 对图像')

    root_out = args.root_A.rstrip('/\\') + '_crop'
   
    # 统计成功/失败数
    ok_cnt, err_cnt = 0, 0
    
    for idx, (cln, hzy) in enumerate(tqdm(pairs, desc="Processing"), 1):
        try:
            rel = relpath(cln, args.root_A)
            out_cln = join(root_out, rel)
            out_hzy = join(root_out, splitext(rel)[0] + '_hazy_.jpg')
            os.makedirs(dirname(out_cln), exist_ok=True)
    
            warp_hazy, mask = register_once(model, detector, device, cln, hzy, args)
            if warp_hazy is None:
                tqdm.write(f'[{idx}/{len(pairs)}] 配准失败，跳过：{rel}')
                err_cnt += 1
                continue
    
            raw = read_image(cln)
            raw, _ = preprocess(raw, resize_max=args.resize_max)
            raw = (raw.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[..., ::-1]
    
            raw_crop, _ = crop_valid(raw, mask)
            hzy_crop, _ = crop_valid(warp_hazy, mask)
    
            cv2.imwrite(out_cln, raw_crop)
            cv2.imwrite(out_hzy, hzy_crop)
            #tqdm.write(f'[{idx}/{len(pairs)}] 已保存 → {out_cln}  &  {out_hzy}')
            ok_cnt += 1
    
        except Exception as e:
            err_cnt += 1
            tqdm.write(f'[{idx}/{len(pairs)}] 处理异常，跳过：{rel} | 错误：{e}')
            # traceback.print_exc()   # 如需详细栈信息，可取消注释
    
    tqdm.write(f'全部完成！成功：{ok_cnt} 张，失败：{err_cnt} 张')


if __name__ == '__main__':
    main()
