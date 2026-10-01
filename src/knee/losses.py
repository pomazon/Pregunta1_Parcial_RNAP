"""Pérdidas multietiqueta que aceptan objetivos blandos (0, 0.5, 1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) para sigmoide, con objetivos blandos.

    L = BCE(p, y) * (1 - p_t)^gamma, con p_t = p*y + (1-p)*(1-y).
    Reduce el peso de los ejemplos fáciles (mayoría de negativos claros).
    """

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, target):
        ce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        return (ce * (1 - p_t).pow(self.gamma)).mean()


def build_loss(name, pos_weight=None, gamma=2.0):
    return FocalLoss(gamma, pos_weight) if name == "focal" else BCELoss(pos_weight)
