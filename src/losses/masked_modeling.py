"""
Masked sensor modeling loss: reconstruct (mean, std) statistics of masked
patches from the Transformer's contextual output at those positions.
Predicts summary statistics rather than raw waveforms — reconstructing raw
samples risks the model spending capacity on sensor-specific noise;
(mean, std) still forces real understanding of local signal behavior
without that failure mode. A per-axis or spectral target is a natural
ablation to add later.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 6)  # per-axis (mean, std) x up to 3 axes

    def forward(self, token_reprs: torch.Tensor) -> torch.Tensor:
        return self.proj(token_reprs)


def compute_msm_loss(head: MSMHead, token_reprs: torch.Tensor, mm_targets: torch.Tensor,
                      mm_mask: torch.Tensor) -> torch.Tensor:
    real_tokens = token_reprs[:, 1:, :]  # drop CLS
    preds = head(real_tokens)

    if mm_mask.sum() == 0:
        return preds.sum() * 0.0

    preds_masked = preds[mm_mask]
    targets_masked = mm_targets[mm_mask]
    return F.mse_loss(preds_masked, targets_masked)
