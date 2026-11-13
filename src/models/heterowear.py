"""
HeteroWearEncoder: hierarchical sensor-aware representation model.

Decomposes representation learning into two stages, rather than
concatenating every channel's patches into one flat sequence (the
original HW-Flat design, kept in hw_flat.py for
comparison):

  1. Within-sensor temporal encoding: each present sensor's own patch
     sequence is processed independently by a modality-shared temporal
     Transformer, then pooled to one sensor-level embedding. Sensor
     provenance (modality/location/device/rate) is added AFTER this
     pooling, not per-patch — patches only carry a temporal-position
     embedding.
  2. Cross-sensor relational encoding: the pooled, context-embedded
     sensor representations for a window are processed jointly by a
     second Transformer, then a small set of learned latent queries
     cross-attends to the available sensors to produce a fixed-size
     window representation, regardless of how many sensors were
     present.

A missing sensor contributes no tokens at any stage (no imputation);
padding masks at both the temporal-patch level (implicit, since only
present sensors are tokenized at all) and the cross-sensor level (over
variable sensor-set size S) are what make this a first-class input
rather than a special case.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import torch
import torch.nn as nn

from src.models.hw_flat import ChannelInput  # reused as SensorInput
from src.models.tokenizer import MultiModalTokenizer, MODALITY_N_AXES
from src.models.embeddings import MODALITY_VOCAB, LOCATION_VOCAB, DEVICE_VOCAB, SamplingRateEmbedding

SensorInput = ChannelInput  # alias matching the spec's naming


class TemporalEncoder(nn.Module):
    """Modality-shared per-sensor temporal Transformer. One instance per
    modality (like the tokenizer stems), applied to a single sensor's own
    patch sequence before any cross-sensor information exchange."""

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, P, d_model] -> [B, P, d_model]"""
        return self.encoder(x)


class SensorContextEmbedding(nn.Module):
    """Adds modality/location/device/sampling-rate context to a POOLED
    sensor-level embedding (not per-patch, unlike HW-Flat's
    ContextEmbedding) — provenance is attached once the temporal encoder
    has already summarized that sensor's own dynamics."""

    def __init__(self, d_model: int):
        super().__init__()
        self.modality_emb = nn.Embedding(len(MODALITY_VOCAB), d_model)
        self.location_emb = nn.Embedding(len(LOCATION_VOCAB), d_model)
        self.device_emb = nn.Embedding(len(DEVICE_VOCAB), d_model)
        self.sampling_emb = SamplingRateEmbedding(d_model)

    def forward(self, h_s: torch.Tensor, modality: str, location: str, device: str,
                native_hz: torch.Tensor) -> torch.Tensor:
        """h_s: [B, d_model] pooled sensor embedding. native_hz: [B]. Returns [B, d_model]."""
        dev = h_s.device
        B = h_s.shape[0]
        mod_idx = torch.full((B,), MODALITY_VOCAB[modality], device=dev, dtype=torch.long)
        loc_idx = torch.full((B,), LOCATION_VOCAB[location], device=dev, dtype=torch.long)
        dev_idx = torch.full((B,), DEVICE_VOCAB[device], device=dev, dtype=torch.long)
        ctx = (self.modality_emb(mod_idx) + self.location_emb(loc_idx) + self.device_emb(dev_idx)
               + self.sampling_emb(native_hz))
        return h_s + ctx


class CrossSensorEncoder(nn.Module):
    """Models relationships among the S sensors present in a window
    (redundancy between phone/watch accelerometers, complementary motion
    across body locations, etc.) via a padding-masked Transformer over
    the variable-length sensor set."""

    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, U: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """U: [B, S_max, d_model]. key_padding_mask: [B, S_max], True = padding."""
        return self.encoder(U, src_key_padding_mask=key_padding_mask)


class LatentFusion(nn.Module):
    """K learned latent queries cross-attend to the available (padding-
    masked) sensor representations, producing a fixed-dimensional window
    representation whose cost depends only weakly on how many sensors
    were actually present."""

    def __init__(self, d_model: int, n_latent: int = 4, latent_layers: int = 1,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.zeros(n_latent, d_model))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True, norm_first=True)
        self.latent_self_attn = nn.TransformerEncoder(layer, latent_layers)
        self.n_latent = n_latent

    def forward(self, V: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """V: [B, S_max, d_model]. key_padding_mask: [B, S_max], True = padding.
        Returns g: [B, d_model]."""
        B = V.shape[0]
        Q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, d_model]
        Z, _ = self.cross_attn(Q, V, V, key_padding_mask=key_padding_mask)
        Z = self.latent_self_attn(Z)  # [B, K, d_model]
        return Z.mean(dim=1)


class HeteroWearEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_patches: int = 8, samples_per_patch: int = 16,
                 temporal_layers: int = 2, temporal_heads: int = 4,
                 cross_sensor_layers: int = 4, cross_sensor_heads: int = 4,
                 n_latent_queries: int = 4, latent_layers: int = 1, latent_heads: int = 4,
                 dropout: float = 0.1, use_context: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.use_context = use_context  # ablation knob: skip sensor context embeddings entirely

        self.tokenizer = MultiModalTokenizer(d_model, n_patches, samples_per_patch)
        self.time_emb = nn.Embedding(n_patches, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.temporal_encoders = nn.ModuleDict({
            mod.value: TemporalEncoder(d_model, temporal_layers, temporal_heads, dropout)
            for mod in MODALITY_N_AXES.keys()
        })
        self.context_emb = SensorContextEmbedding(d_model)
        self.cross_sensor_encoder = CrossSensorEncoder(d_model, cross_sensor_layers,
                                                         cross_sensor_heads, dropout)
        self.latent_fusion = LatentFusion(d_model, n_latent_queries, latent_layers,
                                           latent_heads, dropout)

    def forward(self, batch: list[list[SensorInput]], mask_ratio: float = 0.0,
                rng: random.Random | None = None):
        """batch: list of windows, each a list of present SensorInput objects
        (already post structured-dropout, if applicable — a dropped sensor
        simply isn't in this list, so it contributes no tokens anywhere).

        Returns:
            g: [B, d_model] pooled window representation.
            msm_reprs, msm_targets: [N_masked, d_model], [N_masked, 6] flat
                tensors gathered across every masked patch in the batch
                (only when mask_ratio > 0, else both None).
        """
        rng = rng or random
        device = next(self.parameters()).device
        B = len(batch)
        P = self.n_patches

        per_window_u = []  # list of [S_i, d_model] tensors
        msm_reprs, msm_targets = [], []

        for window in batch:
            u_list = []
            for s in window:
                x = s.signal.unsqueeze(0).to(device)  # [1, T, n_axes]
                if mask_ratio > 0:
                    tokens, targets = self.tokenizer(s.modality, x, return_targets=True)
                else:
                    tokens = self.tokenizer(s.modality, x, return_targets=False)
                    targets = None

                time_idx = torch.arange(P, device=device)
                tokens = tokens + self.time_emb(time_idx).unsqueeze(0)  # [1, P, d_model]

                if mask_ratio > 0:
                    n_mask = max(1, int(round(P * mask_ratio)))
                    idx = torch.randperm(P, device=device)[:n_mask]
                    tokens = tokens.clone()
                    tokens[0, idx] = self.mask_token
                else:
                    idx = None

                H_s = self.temporal_encoders[s.modality](tokens)  # [1, P, d_model]

                if mask_ratio > 0:
                    for j in idx.tolist():
                        msm_reprs.append(H_s[0, j])
                        msm_targets.append(targets[0, j])

                h_s = H_s.mean(dim=1)  # [1, d_model]
                if self.use_context:
                    native_hz = torch.tensor([s.native_hz], device=device, dtype=torch.float32)
                    u_s = self.context_emb(h_s, s.modality, s.location, s.device, native_hz)
                else:
                    u_s = h_s
                u_list.append(u_s.squeeze(0))

            if u_list:
                per_window_u.append(torch.stack(u_list))  # [S_i, d_model]
            else:
                per_window_u.append(torch.zeros(0, self.d_model, device=device))

        S_max = max(1, max(u.shape[0] for u in per_window_u))
        U = torch.zeros(B, S_max, self.d_model, device=device)
        key_padding_mask = torch.ones(B, S_max, dtype=torch.bool, device=device)
        for i, u in enumerate(per_window_u):
            n = u.shape[0]
            if n > 0:
                U[i, :n] = u
                key_padding_mask[i, :n] = False

        V = self.cross_sensor_encoder(U, key_padding_mask)
        g = self.latent_fusion(V, key_padding_mask)

        if mask_ratio > 0 and msm_reprs:
            return g, torch.stack(msm_reprs), torch.stack(msm_targets)
        return g, None, None
