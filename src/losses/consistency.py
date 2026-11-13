"""
Cross-configuration consistency loss: two different sensor-subset views of
the SAME window (from structured_dropout) should map to similar
representations; different windows should not. Symmetric InfoNCE, same
formulation as SimCLR — the two "augmented views" are sensor-configuration
subsets rather than image crops.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def cross_config_consistency_loss(pooled_a: torch.Tensor, pooled_b: torch.Tensor,
                                    temperature: float = 0.1) -> torch.Tensor:
    a = F.normalize(pooled_a, dim=-1)
    b = F.normalize(pooled_b, dim=-1)
    logits_ab = a @ b.t() / temperature
    logits_ba = b @ a.t() / temperature
    targets = torch.arange(a.shape[0], device=a.device)
    loss_ab = F.cross_entropy(logits_ab, targets)
    loss_ba = F.cross_entropy(logits_ba, targets)
    return (loss_ab + loss_ba) / 2
