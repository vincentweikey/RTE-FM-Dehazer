#!/usr/bin/env python3
import os
import argparse
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from models.unet import EfficientUNet 
from flow import FlowMatcher
from dataloader import PairedLatentDataset


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(model, opt, epoch, ckpt_dir, name=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    name = name or f"epoch{epoch:03d}.pt"
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch},
               os.path.join(ckpt_dir, name))
    print(f"[save] {name}")


from tqdm import tqdm

def train_one_epoch(model, loader, opt, device, epoch, rank):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, disable=rank != 0, ncols=100, desc="Train")
    for step, batch in enumerate(pbar):
        hazy_latent = batch["hazy_latent"].to(device, non_blocking=True)
        clean_latent = batch["clean_latent"].to(device, non_blocking=True)
        opt.zero_grad()
        loss = model.training_losses(x1=clean_latent, x0=hazy_latent)
        loss.backward()
        opt.step()
        total_loss += loss.item() * hazy_latent.size(0)

        # 实时显示
        if rank == 0:
            lr = opt.param_groups[0]["lr"]
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, epoch, rank):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, disable=rank != 0, ncols=100, desc="Valid")
    for batch in pbar:
        hazy_latent = batch["hazy_latent"].to(device, non_blocking=True)
        clean_latent = batch["clean_latent"].to(device, non_blocking=True)
        loss = model.training_losses(x1=clean_latent, x0=hazy_latent)
        total_loss += loss.item() * hazy_latent.size(0)
        if rank == 0:
            pbar.set_postfix({"valid_loss": f"{loss.item():.4f}"})
    return total_loss / len(loader.dataset)

def main_worker(rank, world_size, args):
    # 1. environment
    set_seed(args.seed + rank)
    if args.multi_gpu:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 2. data
    train_ds = PairedLatentDataset(
        raw_root=os.path.join(args.data_root,'RAW_512'),
        latent_root=os.path.join(args.data_root,'Latent'),
        img_size=64,
        split="train",
        train_ratio=1.0,
        seed=args.seed
    )
    val_ds = PairedLatentDataset(
        raw_root= os.path.join(args.data_root,'RAW_512'),
        latent_root= os.path.join(args.data_root,'Latent'),
        img_size=64,
        split="val",
        train_ratio=0.99,
        seed=args.seed
    )
    sampler = DistributedSampler(train_ds, shuffle=True) if args.multi_gpu else None
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=(sampler is None),
                              sampler=sampler,
                              num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True)

    # 3. model
    unet = EfficientUNet(
        in_channels=4,
        model_channels=args.base_ch,
        out_channels=4,
        num_res_blocks=args.num_res,
        attention_resolutions=args.attn_res,
        dropout=args.drop,
        channel_mult=args.ch_mult,
        use_scale_shift_norm=True
    ).to(device)
    model = FlowMatcher(unet).to(device)
    if args.multi_gpu:
        model = DDP(model, device_ids=[device])

    # 4. optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    # 5. resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] epoch {start_epoch}")

    # 6. TensorBoard
    writer = None
    if rank == 0:
        writer = SummaryWriter(args.ckpt_dir)

    # 7. training loop
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, opt, device, epoch, rank)
        val_loss   = validate(model, val_loader, device, epoch, rank)
        scheduler.step()

        # log & save
        if rank == 0:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"Epoch[{epoch:03d}/{args.epochs}]  train={train_loss:.6f}  val={val_loss:.6f}")
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_ckpt(model if not args.multi_gpu else model.module,
                          opt, epoch, args.ckpt_dir)

    if args.multi_gpu:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="father floder，contain RAW_512/ & Latent/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ckpt_dir", default="/root/autodl-tmp/ckpts_rte64/")
    parser.add_argument("--resume", help="ckpt path")
    parser.add_argument("--multi_gpu", action="store_true", help="USING DDP!")
    parser.add_argument("--save_every", type=int, default=25, help="save every N epoch")
    parser.add_argument("--seed", type=int, default=42)
    # net
    parser.add_argument("--base_ch", type=int, default=128)
    parser.add_argument("--num_res", type=int, default=2)
    parser.add_argument("--attn_res", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    parser.add_argument("--drop", type=float, default=0.1)
    args = parser.parse_args()

    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
