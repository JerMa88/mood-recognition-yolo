from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace) classification head.

    Expects L2-normalized embeddings as input.
    """

    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize class weights
        W = F.normalize(self.weight, dim=1)
        # Cosine similarity between embeddings and weights
        cosine = F.linear(emb, W)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        # Apply margin only to correct classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        # Use torch.where to ensure numerical stability
        logits = torch.where(cosine > self.th, phi, cosine - self.mm)
        logits = (one_hot * logits) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s
        return logits

