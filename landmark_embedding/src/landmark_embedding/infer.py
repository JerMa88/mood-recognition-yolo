from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .data import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class InferConfig:
    data_dir: str
    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 4


def build_infer_loader(cfg: InferConfig) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.Resize(int(cfg.img_size * 1.14)),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    ds = datasets.ImageFolder(root=cfg.data_dir, transform=tf)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )


@torch.no_grad() 
def compute_embeddings(model, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    feats = []
    labels = []
    paths: List[str] = []
    for imgs, lbls in tqdm(loader, desc="Embed", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs).detach().cpu()
        feats.append(emb)
        labels.append(lbls)
    # Collect paths from loader's dataset
    # torchvision ImageFolder returns samples list of (path, class_idx)
    paths = [p for p, _ in loader.dataset.samples]  # type: ignore[attr-defined]
    return torch.cat(feats), torch.cat(labels), paths
