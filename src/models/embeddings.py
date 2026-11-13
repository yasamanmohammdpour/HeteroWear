"""
Context embeddings: z_i = E_signal(x_i) + E_modality + E_location + E_device
                         + E_sampling(f_i) + E_time(t_i)

This is what lets the model distinguish "ACC / wrist / smartwatch / 32Hz"
from "ACC / chest / research IMU / 100Hz" instead of treating every
accelerometer stream identically — the whole point of sensor-aware
tokenization for heterogeneous devices.

Sampling rate is embedded continuously (log-Hz through a small MLP)
rather than via a lookup table, because real sampling rates vary
continuously across datasets/subjects (e.g. WISDM's implied_hz differs
per subject due to real timestamp jitter) — a fixed vocabulary would
either miss values or need constant expansion.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from src.data.schema import Modality, BodyLocation, DeviceClass

MODALITY_VOCAB = {m.value: i for i, m in enumerate(Modality)}
LOCATION_VOCAB = {l.value: i for i, l in enumerate(BodyLocation)}
DEVICE_VOCAB = {d.value: i for i, d in enumerate(DeviceClass)}


class SamplingRateEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, hz: torch.Tensor) -> torch.Tensor:
        """hz: [B] raw Hz values -> [B, d_model]"""
        log_hz = torch.log1p(hz).unsqueeze(-1)  # [B, 1]
        return self.mlp(log_hz)


class ContextEmbedding(nn.Module):
    def __init__(self, d_model: int, n_patches: int):
        super().__init__()
        self.modality_emb = nn.Embedding(len(MODALITY_VOCAB), d_model)
        self.location_emb = nn.Embedding(len(LOCATION_VOCAB), d_model)
        self.device_emb = nn.Embedding(len(DEVICE_VOCAB), d_model)
        self.sampling_emb = SamplingRateEmbedding(d_model)
        self.time_emb = nn.Embedding(n_patches, d_model)  # patch position within window
        self.d_model = d_model

    def forward(self, patch_tokens: torch.Tensor, modality: str, location: str, device: str,
                native_hz: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: [B, n_patches, d_model] (output of a PatchStem for one channel)
        modality/location/device: single string, shared across the whole batch call
            (one call per real channel; batch dim covers different windows)
        native_hz: [B] per-window sampling rate for this channel
        Returns: [B, n_patches, d_model] with context added.
        """
        B, n_patches, _ = patch_tokens.shape
        dev = patch_tokens.device

        mod_idx = torch.full((B,), MODALITY_VOCAB[modality], device=dev, dtype=torch.long)
        loc_idx = torch.full((B,), LOCATION_VOCAB[location], device=dev, dtype=torch.long)
        dev_idx = torch.full((B,), DEVICE_VOCAB[device], device=dev, dtype=torch.long)

        ctx = (self.modality_emb(mod_idx) + self.location_emb(loc_idx) + self.device_emb(dev_idx)
               + self.sampling_emb(native_hz))  # [B, d_model]
        ctx = ctx.unsqueeze(1)  # [B, 1, d_model] — broadcasts across patches

        time_idx = torch.arange(n_patches, device=dev)
        time_ctx = self.time_emb(time_idx).unsqueeze(0)  # [1, n_patches, d_model]

        return patch_tokens + ctx + time_ctx
