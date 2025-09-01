from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_dir: str
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, int]:
    train_tf, val_tf = build_transforms(cfg.img_size)
    train_ds = datasets.ImageFolder(root=f"{cfg.data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(root=f"{cfg.data_dir}/val", transform=val_tf)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes

