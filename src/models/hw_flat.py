"""
HWFlatEncoder: the full sensor-aware representation model.

Handles the core heterogeneity problem directly: different windows have
different SETS of present channels (missing modalities), and different
NUMBERS of raw samples per channel. Each window's present channels are
independently tokenized (per-modality stem) and context-embedded, then
concatenated into one token sequence and padded to the batch max — the
padding mask ensures the Transformer never attends to a channel that
isn't actually present in a given window, which is what makes "missing
modality" a first-class input rather than a special case.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn

from src.models.tokenizer import MultiModalTokenizer
from src.models.embeddings import ContextEmbedding
from src.models.transformer import SharedTransformerEncoder


@dataclass
class ChannelInput:
    modality: str
    location: str
    device: str
    native_hz: float
    signal: torch.Tensor  # [T, n_axes], T may vary


def window_dict_to_channel_inputs(window_dict: dict) -> list[ChannelInput]:
    """Parses one serialized window dict (as loaded from a 04_export_raw_all_subjects.py
    pickle file) into model-ready ChannelInput objects."""
    out = []
    for k in window_dict["channel_keys"]:
        modality, location, device, native_hz, n_axes, is_derived = window_dict[f"meta__{k}"]
        arr = np.asarray(window_dict[f"sig__{k}"], dtype=np.float32)
        sig = torch.from_numpy(arr)
        if sig.ndim == 1:
            sig = sig.unsqueeze(-1)
        out.append(ChannelInput(modality=modality, location=location, device=device,
                                  native_hz=float(native_hz), signal=sig))
    return out


class HWFlatEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_patches: int = 8, samples_per_patch: int = 16,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.tokenizer = MultiModalTokenizer(d_model, n_patches, samples_per_patch)
        self.context = ContextEmbedding(d_model, n_patches)
        self.transformer = SharedTransformerEncoder(d_model, n_layers, n_heads, dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, batch: list[list[ChannelInput]], mask_ratio: float = 0.0):
        """
        batch: one list of ChannelInput per window (variable-length lists —
               this is exactly where "missing modality" shows up as input).
        mask_ratio: if > 0, randomly replaces this fraction of REAL (non-padding)
            patch tokens with a learnable mask embedding before the Transformer,
            and returns the (mean, std) reconstruction target for those positions.

        Returns:
            pooled: [B, d_model]
            token_reprs: [B, N_max+1, d_model]  (CLS + padded tokens)
            padding_mask: [B, N_max]  bool, True = padding
            mm_targets: [B, N_max, 2]  (mean, std) target per position
            mm_mask: [B, N_max]  bool, True where masked
        """
        device = next(self.parameters()).device
        per_window_tokens = []
        per_window_targets = []
        for channels in batch:
            if not channels:
                per_window_tokens.append(torch.zeros(0, self.d_model, device=device))
                per_window_targets.append(torch.zeros(0, 2, device=device))
                continue
            blocks, target_blocks = [], []
            for ch in channels:
                sig = ch.signal.to(device).unsqueeze(0)  # [1, T, n_axes]
                hz = torch.tensor([ch.native_hz], device=device, dtype=torch.float32)
                tok, tgt = self.tokenizer(ch.modality, sig, return_targets=True)
                tok = self.context(tok, ch.modality, ch.location, ch.device, hz)
                blocks.append(tok.squeeze(0))
                target_blocks.append(tgt.squeeze(0))
            per_window_tokens.append(torch.cat(blocks, dim=0))
            per_window_targets.append(torch.cat(target_blocks, dim=0))

        max_n = max(t.shape[0] for t in per_window_tokens)
        max_n = max(max_n, 1)
        B = len(per_window_tokens)
        padded = torch.zeros(B, max_n, self.d_model, device=device)
        padding_mask = torch.ones(B, max_n, dtype=torch.bool, device=device)
        mm_targets = torch.zeros(B, max_n, 6, device=device)
        mm_mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)

        for i, (t, tgt) in enumerate(zip(per_window_tokens, per_window_targets)):
            n = t.shape[0]
            if n == 0:
                continue
            padded[i, :n] = t
            padding_mask[i, :n] = False
            mm_targets[i, :n] = tgt
            if mask_ratio > 0 and n > 1:
                n_mask = max(1, int(round(n * mask_ratio)))
                mask_idx = torch.randperm(n, device=device)[:n_mask]
                padded[i, mask_idx] = self.mask_token.squeeze(0)
                mm_mask[i, mask_idx] = True

        pooled, token_reprs = self.transformer(padded, padding_mask)
        return pooled, token_reprs, padding_mask, mm_targets, mm_mask
