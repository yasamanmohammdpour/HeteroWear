"""
Wraps HWFlatEncoder with a classification head for fine-tuning —
unlike linear probing (frozen encoder, single forward pass, sklearn head),
this backpropagates through the whole encoder, letting pretrained features
adapt to the specific task instead of just being read out linearly.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from src.models.hw_flat import HWFlatEncoder


class HWFlatClassifier(nn.Module):
    def __init__(self, encoder: HWFlatEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, n_classes)

    def forward(self, batch):
        pooled, _, _, _, _ = self.encoder(batch, mask_ratio=0.0)
        return self.head(pooled)


def load_pretrained_encoder(checkpoint_path, device: str) -> HWFlatEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    encoder = HWFlatEncoder(
        d_model=args["d_model"], n_patches=args["n_patches"],
        samples_per_patch=args["samples_per_patch"],
        n_layers=args["n_layers"], n_heads=args["n_heads"],
    ).to(device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    return encoder, ckpt["step"]
