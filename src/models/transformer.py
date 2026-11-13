"""
Shared Transformer encoder — takes the concatenated set of context-embedded
patch tokens from every present channel in a window (variable count, since
missing modalities mean fewer channels -> fewer tokens) and produces a
pooled window representation via a learnable CLS token.

Standard pre-LN transformer blocks; nothing exotic here on purpose — the
sensor-aware tokenization and context embeddings are where the real
architectural decisions live, not the backbone.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, N, d_model], key_padding_mask: [B, N] True = PADDING (ignored)"""
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SharedTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor):
        """
        tokens: [B, N, d_model] — all channels' patch tokens concatenated, PADDED to max N in batch
        padding_mask: [B, N] bool, True where tokens[b, n] is padding (not a real sensor patch)
        Returns:
            pooled: [B, d_model]  (CLS token representation — the window-level embedding)
            token_reprs: [B, N+1, d_model]  (per-token representations, CLS prepended;
                used later for masked-sensor-modeling reconstruction targets)
        """
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, tokens], dim=1)  # [B, N+1, d_model]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)  # CLS is never padding
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, N+1]

        for block in self.blocks:
            x = block(x, key_padding_mask=full_mask)
        x = self.final_norm(x)

        pooled = x[:, 0]  # CLS token
        return pooled, x
