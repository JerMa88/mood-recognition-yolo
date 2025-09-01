from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_dir: str
    img_size: int = 48
    batch_size: int = 256
    num_workers: int = 4
    grayscale: bool = True
    weighted_sampler: bool = False


def build_transforms(img_size: int, grayscale: bool = True):
    tfs = []
    if grayscale:
        tfs.append(transforms.Grayscale(num_output_channels=1))
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            *tfs,
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        *tfs,
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def compute_class_weights(ds: datasets.ImageFolder) -> torch.Tensor:
    counts = torch.zeros(len(ds.classes), dtype=torch.float)
    for _, y in ds.samples:
        counts[y] += 1
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.sum() * len(ds.classes)
    return weights


def build_dataloaders(
    cfg: DataConfig,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, int, Optional[DistributedSampler], Optional[DistributedSampler]]:
    train_tf, val_tf = build_transforms(cfg.img_size, cfg.grayscale)
    train_ds = datasets.ImageFolder(root=f"{cfg.data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(root=f"{cfg.data_dir}/val", transform=val_tf)
    num_classes = len(train_ds.classes)

    train_sampler_d: Optional[DistributedSampler] = None
    val_sampler_d: Optional[DistributedSampler] = None
    sample_sampler = None

    if distributed:
        train_sampler_d = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler_d = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    if cfg.weighted_sampler and not distributed:
        class_weights = compute_class_weights(train_ds)
        sample_weights = torch.tensor([class_weights[y] for _, y in train_ds.samples], dtype=torch.float)
        sample_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler_d is None and sample_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=sample_sampler if sample_sampler is not None else train_sampler_d,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=val_sampler_d,
    )
    return train_loader, val_loader, num_classes, train_sampler_d, val_sampler_d
