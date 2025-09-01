from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .data import DataConfig, build_dataloaders
from .losses import ArcFaceHead
from .models import create_model


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "runs/exp"
    arch: str = "resnet50"
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 512
    pretrained: bool = True
    freeze_backbone: bool = False
    num_workers: int = 4
    amp: bool = True
    seed: int = 0


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> Tuple[torch.Tensor, ...]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return tuple(res)


def train_one_epoch(
    model, head, loader, optimizer, scaler, device, epoch, amp=True
):
    model.train()
    head.train()
    ce = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
    running_loss = 0.0
    running_top1 = 0.0
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            emb = model(imgs)
            logits = head(emb, labels)
            loss = ce(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        top1 = accuracy(logits.detach(), labels, topk=(1,))[0]
        running_loss += loss.item() * imgs.size(0)
        running_top1 += top1.item() * imgs.size(0) / 100.0
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{top1.item():.2f}"})

    n = len(loader.dataset)
    return running_loss / n, 100.0 * running_top1 / n


@torch.no_grad()
def validate(model, head, loader, device, amp=True):
    model.eval()
    head.eval()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_top1 = 0.0
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=amp):
            emb = model(imgs)
            # For eval, we still need logits; pass labels to ArcFace to get margins applied
            logits = head(emb, labels)
            loss = ce(logits, labels)
        top1 = accuracy(logits, labels, topk=(1,))[0]
        running_loss += loss.item() * imgs.size(0)
        running_top1 += top1.item() * imgs.size(0) / 100.0
    n = len(loader.dataset)
    return running_loss / n, 100.0 * running_top1 / n


def save_checkpoint(state: dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, "last.pt"))
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pt"))


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train landmark embedding with ArcFace")
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/ and val/")
    parser.add_argument("--out-dir", default="runs/exp", help="Output directory for checkpoints")
    parser.add_argument("--arch", default="resnet50", help="timm backbone name")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parsed = parser.parse_args(args)

    cfg = TrainConfig(
        data_dir=parsed.data_dir,
        out_dir=parsed.out_dir,
        arch=parsed.arch,
        img_size=parsed.img_size,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        embedding_dim=parsed.embedding_dim,
        pretrained=not parsed.no_pretrained,
        freeze_backbone=parsed.freeze_backbone,
        num_workers=parsed.num_workers,
        amp=not parsed.no_amp,
        seed=parsed.seed,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, num_classes = build_dataloaders(
        DataConfig(cfg.data_dir, cfg.img_size, cfg.batch_size, cfg.num_workers)
    )

    # Model & head
    model = create_model(
        arch=cfg.arch,
        embedding_dim=cfg.embedding_dim,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)
    head = ArcFaceHead(cfg.embedding_dim, num_classes).to(device)

    # Optim & sched
    opt = optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr},
            {"params": head.parameters(), "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)

    best_top1 = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_top1 = train_one_epoch(model, head, train_loader, opt, scaler, device, epoch, cfg.amp)
        val_loss, val_top1 = validate(model, head, val_loader, device, cfg.amp)
        sched.step()

        is_best = val_top1 > best_top1
        best_top1 = max(best_top1, val_top1)

        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "best_top1": best_top1,
                "cfg": cfg.__dict__,
                "num_classes": num_classes,
            },
            is_best,
            cfg.out_dir,
        )

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} top1 {train_top1:.2f} | "
            f"val loss {val_loss:.4f} top1 {val_top1:.2f} | best {best_top1:.2f}"
        )


if __name__ == "__main__":
    main()

