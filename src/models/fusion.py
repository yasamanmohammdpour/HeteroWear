"""
FusionClassifier: combines HW-Flat's pretrained multimodal representation
with a supervised single-channel TSTransformer backbone via cross-attention
and a learned per-example, per-feature gate — rather than naive
concatenation.

Motivation, directly from the observed result pattern: HW-Flat's SSL
representation decisively wins on cross-device heterogeneity (HHAR) but
loses to plain supervised training on fine-grained or noisy-label tasks
(WISDM, HARTH). A fixed concatenation can't adapt to that asymmetry across
datasets, or even across individual examples within one dataset. A learned
gate can: it lets the network decide, per example, how much to trust each
source, and cross-attention lets each representation be refined in light
of what the other one saw before that decision is made.

Design:
  1. Project both pooled embeddings to a common fusion dimension.
  2. Stack as a 2-token sequence with a learned source embedding (so the
     attention layer knows which token came from which source), and apply
     a small self-attention stack so each token can incorporate
     information from the other before the gate reads them.
  3. A gating MLP looks at BOTH pre-attention projections and outputs a
     per-feature weight in [0, 1] indicating how much to trust the
     HW-Flat-derived token vs. the supervised-backbone-derived token
     for this specific example.
  4. The fused representation is the gated combination of the two
     (post-attention) tokens, fed to a final linear classifier.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from src.models.hw_flat import HWFlatEncoder
from src.models_baseline.ts_transformer import TSTransformer


class FusionBlock(nn.Module):
    def __init__(self, d_fusion: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.source_emb = nn.Embedding(2, d_fusion)  # 0 = HW-Flat token, 1 = supervised token
        layer = nn.TransformerEncoderLayer(d_fusion, n_heads, dim_feedforward=d_fusion * 4,
                                            dropout=dropout, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, n_layers)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_fusion * 2, d_fusion), nn.GELU(), nn.Linear(d_fusion, d_fusion), nn.Sigmoid(),
        )

    def forward(self, p_cw: torch.Tensor, p_tf: torch.Tensor) -> torch.Tensor:
        """p_cw, p_tf: [B, d_fusion] projected pooled embeddings. Returns
        [B, d_fusion] fused representation."""
        B = p_cw.shape[0]
        idx = torch.arange(2, device=p_cw.device)
        src = self.source_emb(idx).unsqueeze(0)  # [1, 2, d_fusion]

        tokens = torch.stack([p_cw, p_tf], dim=1) + src  # [B, 2, d_fusion]
        attended = self.attn(tokens)  # [B, 2, d_fusion]
        attn_cw, attn_tf = attended[:, 0], attended[:, 1]

        gate = self.gate_mlp(torch.cat([p_cw, p_tf], dim=-1))  # [B, d_fusion], in (0,1)
        fused = gate * attn_cw + (1 - gate) * attn_tf
        return fused


class FusionClassifier(nn.Module):
    def __init__(self, flat_encoder: HWFlatEncoder, n_channels: int, n_classes: int,
                 d_fusion: int = 128, tf_d_model: int = 64, tf_n_layers: int = 3,
                 fusion_n_layers: int = 2):
        super().__init__()
        self.cw_encoder = flat_encoder
        self.tf_backbone = TSTransformer(n_channels=n_channels, n_classes=n_classes,
                                          d_model=tf_d_model, n_layers=tf_n_layers)
        self.proj_cw = nn.Sequential(nn.Linear(flat_encoder.d_model, d_fusion), nn.LayerNorm(d_fusion))
        self.proj_tf = nn.Sequential(nn.Linear(tf_d_model, d_fusion), nn.LayerNorm(d_fusion))
        self.fusion = FusionBlock(d_fusion, n_layers=fusion_n_layers)
        self.head = nn.Linear(d_fusion, n_classes)

    def forward(self, cw_batch, tf_input: torch.Tensor) -> torch.Tensor:
        """cw_batch: list[list[ChannelInput]] for HWFlatEncoder.
        tf_input: [B, T, C] primary-channel array for the TSTransformer backbone.
        Both must correspond to the SAME windows, in the same order."""
        pooled_cw, _, _, _, _ = self.cw_encoder(cw_batch, mask_ratio=0.0)
        pooled_tf = self.tf_backbone(tf_input, return_embedding=True)

        p_cw = self.proj_cw(pooled_cw)
        p_tf = self.proj_tf(pooled_tf)
        fused = self.fusion(p_cw, p_tf)
        return self.head(fused)
