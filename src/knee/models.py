"""Modelos a nivel de estudio: codificador 2D por corte + agregación entre cortes.

* Codificadores
    - "scratch": CNN propia (5 bloques conv-BN-ReLU x2 + max-pool), entrenada desde cero.
    - nombres de timm ("resnet50", "resnet18", "efficientnet_b0", ...): backbone
      preentrenado en ImageNet, sin su clasificador.
* Agregación (implementación propia)
    - "attn": atención con compuerta por etiqueta (Ilse et al., 2018, extendida a
      multietiqueta): cada hallazgo aprende en qué cortes mirar (p. ej. meniscos en los
      extremos, LCA en el centro). Los pesos sirven también para interpretar.
    - "max" / "mean": agregaciones simples para comparar.
* Salida: 12 logits (sigmoide independiente por etiqueta en la pérdida).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchEncoder(nn.Module):
    """CNN 2D desde cero. 256x256 -> mapa 8x8 x widths[-1]."""

    def __init__(self, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers.append(_block(c, w))
            c = w
        self.features = nn.Sequential(*layers)
        self.out_dim = c
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.features(x)


class TimmEncoder(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        import timm
        self.features = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        self.out_dim = self.features.num_features

    def forward(self, x):
        return self.features(x)


class StudyModel(nn.Module):
    def __init__(self, encoder, n_labels=12, pool="attn", hidden=128, dropout=0.3, imagenet_norm=True):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.n_labels = n_labels
        d = encoder.out_dim
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if imagenet_norm else ((0.5,) * 3, (0.25,) * 3)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.drop = nn.Dropout(dropout)
        if pool == "attn":
            self.att_v = nn.Linear(d, hidden)
            self.att_u = nn.Linear(d, hidden)
            self.att_w = nn.Linear(hidden, n_labels)
            self.cls_w = nn.Parameter(torch.empty(n_labels, d))
            self.cls_b = nn.Parameter(torch.zeros(n_labels))
            nn.init.xavier_uniform_(self.cls_w)
        else:
            self.head = nn.Linear(d, n_labels)

    def slice_features(self, x):
        """x [B, P, 3, H, W] -> mapas [B*P, d, h, w] y vectores [B, P, d]."""
        b, p = x.shape[:2]
        x = (x.flatten(0, 1) - self.mean) / self.std
        fmap = self.encoder(x)
        vec = F.adaptive_avg_pool2d(fmap, 1).flatten(1).view(b, p, -1)
        return fmap, vec

    def aggregate(self, vec):
        """vec [B, P, d] -> logits [B, L], atención [B, P, L] (o None)."""
        if self.pool == "attn":
            a = self.att_w(torch.tanh(self.att_v(vec)) * torch.sigmoid(self.att_u(vec)))  # [B, P, L]
            a = torch.softmax(a, dim=1)
            z = torch.einsum("bpl,bpd->bld", a, vec)                                        # [B, L, d]
            logits = (self.drop(z) * self.cls_w).sum(-1) + self.cls_b
            return logits, a
        z = vec.max(1).values if self.pool == "max" else vec.mean(1)
        return self.head(self.drop(z)), None

    def forward(self, x):
        _, vec = self.slice_features(x)
        logits, _ = self.aggregate(vec)
        return logits


def build_model(name, n_labels=12, pool="attn", pretrained=True):
    if name == "scratch":
        enc = ScratchEncoder()
        return StudyModel(enc, n_labels, pool, imagenet_norm=False)
    enc = TimmEncoder(name, pretrained=pretrained)
    return StudyModel(enc, n_labels, pool, imagenet_norm=True)
