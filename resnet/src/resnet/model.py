from __future__ import annotations

import torch
import torch.nn as nn
import timm


def make_resnet(arch: str = "resnet18", num_classes: int = 7, pretrained: bool = True, in_chans: int = 1) -> nn.Module:
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    return model

