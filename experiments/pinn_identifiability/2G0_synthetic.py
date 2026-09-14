"""GLASSCHIP-PINN Phase 2G0 - Synthetic Ground-Truth Validation."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy.integrate import solve_ivp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from glasschip.models import ClassicalBaselineModel
from glasschip.models.pinn import TemperaturePINN, train_pinn

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_power(t_eval: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(t_eval)
    p_base = np.zeros(n)
    segment_len = max(1, n // 6)
    levels = [50.0, 250.0, 120.0, 320.0, 80.0, 200.0]
    for i, lvl in enumerate(levels):
        s_idx = i * segment_len
        e_idx = (i + 1) * segment_len if i < 5 else n
        p_base[s_idx:e_idx] = lvl
        
    ramps = np.sin(t_eval / 100.0) * 30.0
    noise = rng.normal(0.0, 5.0, size=n)
    return np.clip(p_base + ramps + noise, 20.0, 400.0)


def simulate_synthetic_trajectory(
    tau_true: float,
    K_true: float,
    Ta_val: float = 22.0,
    t_span: tuple[float, float] = (0.0, 1800.0),
    dt: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    P_eval = generate_synthetic_power(t_eval, seed=seed)
    
    def ode_func(t: float, T: list[float]) -> list[float]:
        idx = int(np.clip(t / dt, 0, len(P_eval) - 1))
        P_t = P_eval[idx]
        return [-(1.0 / tau_true) * (T[0] - Ta_val) + K_true * P_t]
        
    sol = solve_ivp(ode_func, t_span, [Ta_val], t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8)
    return t_eval, P_eval, np.full_like(t_eval, Ta_val), sol.y[0]


def run_phase_2g0() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G0: Synthetic Benchmark ===")
    
    tau_targets = [10.0, 30.0, 60.0, 120.0]
    K_targets = [0.05, 0.10]
    
    results_list = []
    clean_passed = True
    
    for tau_true in tau_targets:
        for K_true in K_targets:
            t_ref, P_ref, Ta_ref, T_ref = simulate_synthetic_trajectory(tau_true=tau_true, K_true=K_true)
            
            sub_idx = np.arange(0, len(t_ref), 10)
            t_10s, P_10s, Ta_10s, T_10s = t_ref[sub_idx], P_ref[sub_idx], Ta_ref[sub_idx], T_ref[sub_idx]
            
            ols_fit = ClassicalBaselineModel(dt_s=10.0).fit([(T_10s, P_10s)])
            tau_ols = ols_fit.tau_eff_s
            K_ols = ols_fit.r_eff / tau_ols if tau_ols > 0 else float("nan")
            
            pinn_model = TemperaturePINN(hidden_dim=64, tau_init=tau_ols if np.isfinite(tau_ols) else tau_true, K_init=K_ols if np.isfinite(K_ols) else K_true)
            res_pinn = train_pinn(model=pinn_model, t_obs=t_10s, p_obs=P_10s, ta_obs=Ta_10s, temp_obs=T_10s, epochs=250, lr=3e-3, lam_phys=0.2)
            
            tau_pinn = res_pinn["tau_fitted"]
            K_pinn = res_pinn["K_fitted"]
            
            err_tau_pinn = abs(tau_pinn - tau_true) / tau_true * 100.0
            err_tau_ols = abs(tau_ols - tau_true) / tau_true * 100.0
            
            print(f"Target: tau={tau_true:5.1f}s, K={K_true:.3f} | OLS tau={tau_ols:5.1f}s (err: {err_tau_ols:4.1f}%) | PINN tau={tau_pinn:5.1f}s (err: {err_tau_pinn:4.1f}%)")
            
            if err_tau_pinn > 15.0:
                clean_passed = False
                
            results_list.append({
                "tau_true": tau_true, "K_true": K_true, "tau_ols": tau_ols, "K_ols": K_ols,
                "err_tau_ols_pct": err_tau_ols, "tau_pinn": tau_pinn, "K_pinn": K_pinn, "err_tau_pinn_pct": err_tau_pinn
            })
            
    summary = {
        "phase": "2G0_synthetic",
        "generated": datetime.now(timezone.utc).isoformat(),
        "clean_validation_passed": clean_passed,
        "n_experiments": len(results_list),
        "results": results_list,
    }
    
    out_file = RESULTS_DIR / "phase2g0_synthetic.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G0 complete. Results saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g0()
