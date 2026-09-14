"""Continuous-time Inverse Physics-Informed Neural Network (PINN) for thermal dynamics.

Governing physical ODE:
    dT/dt = -(1/tau)*(T - Ta) + K * P

Parameterization:
    tau = exp(eta_tau)  > 0  [seconds]
    K   = exp(eta_K)    > 0  [K/W]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperaturePINN(nn.Module):
    """Continuous-time inverse PINN for single-component thermal identification."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "tanh",
        tau_init: float = 300.0,
        K_init: float = 0.05,
        t_scale: float = 1800.0,
        p_scale: float = 200.0,
    ) -> None:
        super().__init__()
        
        self.t_scale = t_scale
        self.p_scale = p_scale
        
        act_cls = nn.Tanh if activation.lower() == "tanh" else nn.SiLU
            
        layers: List[nn.Module] = []
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(act_cls())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        self.eta_tau = nn.Parameter(torch.tensor(math.log(max(tau_init, 1e-1)), dtype=torch.float32))
        self.eta_K = nn.Parameter(torch.tensor(math.log(max(K_init, 1e-4)), dtype=torch.float32))
        
    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.eta_tau)

    @property
    def K(self) -> torch.Tensor:
        return torch.exp(self.eta_K)

    def forward(self, t: torch.Tensor, p: torch.Tensor, ta: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.t_scale
        p_norm = p / self.p_scale
        ta_norm = (ta - 20.0) / 10.0
        inputs = torch.cat([t_norm, p_norm, ta_norm], dim=-1)
        return self.net(inputs)

    def physics_residual(
        self, t: torch.Tensor, p: torch.Tensor, ta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
            
        t_hat = self.forward(t, p, ta)
        
        dT_dt = torch.autograd.grad(
            outputs=t_hat,
            inputs=t,
            grad_outputs=torch.ones_like(t_hat),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        residual = dT_dt + (1.0 / self.tau) * (t_hat - ta) - self.K * p
        return t_hat, residual


class HybridTemperaturePINN(nn.Module):
    """Hybrid PINN incorporating a bounded neural discrepancy term g_phi(T, P)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        tau_init: float = 300.0,
        K_init: float = 0.05,
        max_discrepancy: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_pinn = TemperaturePINN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            tau_init=tau_init,
            K_init=K_init,
        )
        self.max_discrepancy = max_discrepancy
        self.g_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    @property
    def tau(self) -> torch.Tensor:
        return self.base_pinn.tau

    @property
    def K(self) -> torch.Tensor:
        return self.base_pinn.K

    def forward(self, t: torch.Tensor, p: torch.Tensor, ta: torch.Tensor) -> torch.Tensor:
        return self.base_pinn(t, p, ta)

    def physics_residual(
        self, t: torch.Tensor, p: torch.Tensor, ta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
            
        t_hat = self.forward(t, p, ta)
        
        dT_dt = torch.autograd.grad(
            outputs=t_hat,
            inputs=t,
            grad_outputs=torch.ones_like(t_hat),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        g_val = self.max_discrepancy * self.g_net(torch.cat([t_hat, p], dim=-1))
        residual = dT_dt + (1.0 / self.tau) * (t_hat - ta) - self.K * p - g_val
        return t_hat, residual


def train_pinn(
    model: nn.Module,
    t_obs: np.ndarray,
    p_obs: np.ndarray,
    ta_obs: np.ndarray,
    temp_obs: np.ndarray,
    t_colloc: Optional[np.ndarray] = None,
    p_colloc: Optional[np.ndarray] = None,
    ta_colloc: Optional[np.ndarray] = None,
    epochs: int = 1500,
    lr: Optional[float] = None,
    lr_net: float = 1e-3,
    lr_params: float = 1e-2,
    lam_data: float = 1.0,
    lam_phys: float = 0.5,
    lam_ic: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Union[float, List[float]]]:
    """Train PINN with backward-compatible learning rate arguments."""
    if lr is not None:
        lr_net = lr

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    p_obs_t = torch.tensor(p_obs, dtype=torch.float32).view(-1, 1)
    ta_obs_t = torch.tensor(ta_obs, dtype=torch.float32).view(-1, 1)
    temp_obs_t = torch.tensor(temp_obs, dtype=torch.float32).view(-1, 1)
    
    if t_colloc is None:
        t_f_t, p_f_t, ta_f_t = t_obs_t, p_obs_t, ta_obs_t
    else:
        t_f_t = torch.tensor(t_colloc, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        p_f_t = torch.tensor(p_colloc, dtype=torch.float32).view(-1, 1)
        ta_f_t = torch.tensor(ta_colloc, dtype=torch.float32).view(-1, 1)

    if hasattr(model, 'net'):
        net_params = model.net.parameters()
        eta_params = [model.eta_tau, model.eta_K]
    else:
        net_params = model.base_pinn.net.parameters()
        eta_params = [model.base_pinn.eta_tau, model.base_pinn.eta_K]
        
    optimizer = optim.Adam([
        {"params": net_params, "lr": lr_net},
        {"params": eta_params, "lr": lr_params},
    ])
    
    history: Dict[str, List[float]] = {
        "loss": [], "loss_data": [], "loss_phys": [], "tau": [], "K": []
    }
    
    t0, p0, ta0, temp0 = t_obs_t[0:1], p_obs_t[0:1], ta_obs_t[0:1], temp_obs_t[0:1]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        t_hat_obs = model(t_obs_t, p_obs_t, ta_obs_t)
        loss_data = torch.mean((t_hat_obs - temp_obs_t) ** 2)
        
        _, residual = model.physics_residual(t_f_t, p_f_t, ta_f_t)
        loss_phys = torch.mean(residual ** 2)
        
        t_hat_ic = model(t0, p0, ta0)
        loss_ic = torch.mean((t_hat_ic - temp0) ** 2)
        
        total_loss = lam_data * loss_data + lam_phys * loss_phys + lam_ic * loss_ic
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        tau_curr = float(model.tau.detach().item())
        K_curr = float(model.K.detach().item())
        
        history["loss"].append(float(total_loss.item()))
        history["loss_data"].append(float(loss_data.item()))
        history["loss_phys"].append(float(loss_phys.item()))
        history["tau"].append(tau_curr)
        history["K"].append(K_curr)
        
    return {
        "tau_fitted": float(model.tau.detach().item()),
        "K_fitted": float(model.K.detach().item()),
        "final_loss": history["loss"][-1],
        "final_loss_data": history["loss_data"][-1],
        "final_loss_phys": history["loss_phys"][-1],
        "history": history,
    }
