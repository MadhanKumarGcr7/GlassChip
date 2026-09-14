"""Standard MLP Baseline for thermal prediction under degraded telemetry."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 1000,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPBaseline(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
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
