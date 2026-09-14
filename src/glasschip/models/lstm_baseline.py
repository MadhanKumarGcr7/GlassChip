"""LSTM Baseline for thermal prediction under degraded telemetry."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        feature_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


def train_lstm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 1000,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMBaseline(feature_dim=X_train.shape[2], hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_tr)
        loss = torch.mean((pred - y_tr) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(x_te).numpy().ravel()
        test_actual = y_te.numpy().ravel()

    rmse = float(np.sqrt(np.mean((test_pred - test_actual) ** 2)))
    ss_res = float(np.sum((test_actual - test_pred) ** 2))
    ss_tot = float(np.sum((test_actual - np.mean(test_actual)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "r2": r2, "predictions": test_pred.tolist()}
