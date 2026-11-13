"""
A compact Transformer time-series classifier — patch embedding + CLS token
+ standard encoder blocks, ViT/PatchTST-style, applied to a single sensor
channel. This is the "SOTA-style" comparison point distinct from
HW-Flat: same general recipe (patchify, attend, classify) but single-
modality, fully supervised, no sensor-aware context embeddings, no
cross-configuration consistency, no missing-modality handling.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score


class TSTransformer(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, patch_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Conv1d(n_channels, d_model, kernel_size=patch_len, stride=patch_len)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 256, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embedding: bool = False, mask_ratio: float = 0.0):
        """x: [B, T, C]. When mask_ratio > 0, masks a fraction of patches and
        returns (pooled, token_reprs, mm_targets, mm_mask) for LIMU-BERT-style
        masked-reconstruction pretraining — same interface as HW-Flat's
        MSMHead/compute_msm_loss, reused directly rather than duplicated.

        Per-instance normalization (zero-mean, unit-std) happens first, the
        same fix already applied to HW-Flat's PatchStem: raw signal
        scale varies enormously across datasets/devices, and without this
        the masked-reconstruction loss is dominated by whichever source in
        a batch has the largest raw scale and never converges (confirmed:
        pretraining without this stayed flat at msm_loss~1.7-1.9 for the
        entire 5000-step run)."""
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-6
        x = (x - mean) / std

        xt = x.transpose(1, 2)
        tokens = self.proj(xt).transpose(1, 2)
        B, N, D = tokens.shape

        mm_targets = mm_mask = None
        if mask_ratio > 0:
            C = xt.shape[1]
            patches = xt.view(B, C, N, self.patch_len)
            patch_mean = patches.mean(dim=3)
            patch_std = patches.std(dim=3)
            if C == 1:
                patch_mean = patch_mean.expand(B, 3, N)
                patch_std = patch_std.expand(B, 3, N)
            elif C != 3:
                patch_mean = patch_mean.mean(dim=1, keepdim=True).expand(B, 3, N)
                patch_std = patch_std.mean(dim=1, keepdim=True).expand(B, 3, N)
            mm_targets = torch.stack([patch_mean, patch_std], dim=-1).permute(0, 2, 1, 3).reshape(B, N, 6)

            tokens = tokens.clone()
            n_mask = max(1, int(round(N * mask_ratio)))
            mm_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            for i in range(B):
                idx = torch.randperm(N, device=x.device)[:n_mask]
                tokens[i, idx] = self.mask_token.squeeze(0)
                mm_mask[i, idx] = True

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1) + self.pos_emb[:, : N + 1]
        out = self.encoder(seq)
        pooled = self.norm(out[:, 0])

        if mask_ratio > 0:
            return pooled, out, mm_targets, mm_mask
        if return_embedding:
            return pooled
        return self.head(pooled)


def run_transformer_baseline(X_train, y_train, X_test, y_test, n_classes: int,
                              epochs: int = 15, batch_size: int = 256, lr: float = 1e-3,
                              device: str = None, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                           batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                          batch_size=batch_size, shuffle=False)

    model = TSTransformer(n_channels=X_train.shape[-1], n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            preds.append(pred); targets.append(yb.numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "n_train": len(y_train), "n_test": len(y_test), "device": device,
    }
