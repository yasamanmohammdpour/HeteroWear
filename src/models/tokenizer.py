"""
Sensor-aware tokenization, stage 1: per-modality patch stems.

Each modality (ACC, GYRO, PPG, ECG, EDA, TEMP, RESP, EMG, HR) gets its own
small encoder, since a 700Hz ECG trace and a 4Hz EDA trace have nothing in
common as raw signals. Each stem:
  1. interpolates the raw [T, n_axes] signal to a canonical length
     (n_patches * samples_per_patch) — robust to the small T variation
     across real windows (coverage thresholds, outlier truncation, etc.)
  2. patchifies via a strided Conv1d into n_patches tokens of dim d_model

Design choice: patches are temporally aligned across modalities (patch i
always corresponds to the same ~1/n_patches slice of the window), which
matters for the cross-configuration consistency objective — two different
sensor subsets of the *same time interval* should produce similar
representations, and that only makes sense if "the same time interval" is
well-defined across modalities.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import Modality

MODALITY_N_AXES = {
    Modality.ACC: 3, Modality.GYRO: 3, Modality.MAG: 3,
    Modality.PPG: 1, Modality.ECG: 1, Modality.EDA: 1,
    Modality.TEMP: 1, Modality.RESP: 1, Modality.EMG: 1, Modality.HR: 1,
}


class PatchStem(nn.Module):
    def __init__(self, n_axes: int, d_model: int, n_patches: int, samples_per_patch: int = 16):
        super().__init__()
        self.n_patches = n_patches
        self.samples_per_patch = samples_per_patch
        self.canonical_len = n_patches * samples_per_patch
        self.proj = nn.Conv1d(n_axes, d_model, kernel_size=samples_per_patch, stride=samples_per_patch)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_targets: bool = False):
        """x: [B, T, n_axes] (T may vary across calls) -> [B, n_patches, d_model]
        If return_targets=True, also returns [B, n_patches, 2] (mean, std) computed
        from the SAME canonical-length interpolated signal the model actually sees —
        used as the reconstruction target for masked sensor modeling."""
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-6
        x = (x - mean) / std

        x = x.transpose(1, 2)  # [B, n_axes, T]
        if x.shape[-1] != self.canonical_len:
            x = F.interpolate(x, size=self.canonical_len, mode="linear", align_corners=False)
        tokens = self.proj(x)  # [B, d_model, n_patches]
        tokens = self.norm(tokens.transpose(1, 2))  # [B, n_patches, d_model]
        if not return_targets:
            return tokens

        B, n_axes, L = x.shape
        patches = x.view(B, n_axes, self.n_patches, self.samples_per_patch)
        patch_mean = patches.mean(dim=3)  # [B, n_axes, n_patches]
        patch_std = patches.std(dim=3)
        if n_axes == 1:
            # single-axis channel: broadcast to all 3 slots, fixed target
            # dim of 6 regardless of modality, no masking needed downstream
            patch_mean = patch_mean.expand(B, 3, self.n_patches)
            patch_std = patch_std.expand(B, 3, self.n_patches)
        # n_axes == 3 (ACC/GYRO/MAG): kept genuinely per-axis rather than
        # pooled — pooling discarded exactly the x/y/z directional detail
        # that distinguishes fine-grained activities sharing similar
        # overall intensity (e.g. WISDM's eating/folding/typing classes).
        targets = torch.stack([patch_mean, patch_std], dim=-1)  # [B, 3, n_patches, 2]
        targets = targets.permute(0, 2, 1, 3).reshape(B, self.n_patches, 6)
        return tokens, targets


class MultiModalTokenizer(nn.Module):
    """One PatchStem per modality, shared across all locations/devices/datasets
    that use that modality — this is what makes the tokenizer dataset-agnostic."""

    def __init__(self, d_model: int, n_patches: int, samples_per_patch: int = 16):
        super().__init__()
        self.stems = nn.ModuleDict({
            mod.value: PatchStem(n_axes, d_model, n_patches, samples_per_patch)
            for mod, n_axes in MODALITY_N_AXES.items()
        })
        self.n_patches = n_patches

    def forward(self, modality: str, x: torch.Tensor, return_targets: bool = False):
        return self.stems[modality](x, return_targets=return_targets)
