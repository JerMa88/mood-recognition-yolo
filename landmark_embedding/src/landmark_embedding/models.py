from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import timm


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=(-2, -1))  # global mean over H,W
        return x.pow(1.0 / self.p)


@dataclass
class BackboneConfig:
    arch: str = "resnet50"
    pretrained: bool = True
    embedding_dim: int = 512
    freeze_backbone: bool = False


class EmbeddingModel(nn.Module):
    """Backbone from timm + GeM pooling + L2-normalized embedding head."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.arch,
            pretrained=cfg.pretrained,
            num_classes=0,  # no classifier
            global_pool="",
            features_only=False,
        )
        feat_dim = self.backbone.num_features
        self.pool = GeM()
        self.fc = nn.Linear(feat_dim, cfg.embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(cfg.embedding_dim)
        self.cfg = cfg

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.ndim == 4:
            feats = self.pool(feats)
        emb = self.fc(feats)
        emb = self.bn(emb)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


def create_model(
    arch: str = "resnet50", embedding_dim: int = 512, pretrained: bool = True, freeze_backbone: bool = False
) -> EmbeddingModel:
    return EmbeddingModel(
        BackboneConfig(
            arch=arch,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone,
        )
    )

