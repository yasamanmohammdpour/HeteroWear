"""
DeepConvLSTM (Ordonez & Roggen, 2016) — the classic strong HAR baseline:
4 conv layers extract local features, 2 LSTM layers capture temporal
dependencies the plain CNN can't. This is the "more robust CNN" reference
point: if HW-Flat's pretrained representation can't beat this on
frozen linear probing, that's a real result worth reporting honestly.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, conv_filters: int = 64,
                 lstm_units: int = 128, n_lstm_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(conv_filters, lstm_units, num_layers=n_lstm_layers,
                             batch_first=True, dropout=0.3 if n_lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_units, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.head(last)


def run_deepconvlstm_baseline(X_train, y_train, X_test, y_test, n_classes: int,
                               epochs: int = 15, batch_size: int = 256, lr: float = 1e-3,
                               device: str = None, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                           batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                          batch_size=batch_size, shuffle=False)

    model = DeepConvLSTM(n_channels=X_train.shape[-1], n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
