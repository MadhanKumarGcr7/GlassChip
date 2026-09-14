"""GLASSCHIP-PINN Phase 2G5 - Combined Degradation Grid & Identifiability Map."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glasschip.models import ClassicalBaselineModel
from glasschip.models.pinn import TemperaturePINN, train_pinn
from pinn_utils import (
    get_pilot_hosts,
    load_or_generate_host_trajectory,
    apply_sparsity_mask,
    quantize_temperature,
)

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_reliability(err_pct: float, ci_width_pct: float, tol_err: float = 10.0, tol_ci: float = 20.0) -> bool:
    return bool((err_pct <= tol_err) and (ci_width_pct <= tol_ci))


def run_phase_2g5() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G5: Combined Identifiability Map ===")
    
    coverages = [1.0, 0.50, 0.25, 0.10, 0.05]
    quant_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    dt_s = 20.0
    
    hosts = get_pilot_hosts()[:2]
    grid_results = []
    
    for cov in coverages:
        for q in quant_levels:
            pinn_taus_seeds = []
            pinn_Ks_seeds = []
            ols_taus_seeds = []
            ols_Ks_seeds = []
            
            for seed in [42, 43]:
                for host in hosts:
                    time_vals, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0, seed=seed)
                    
                    time_sub = time_vals[::2]
                    p_sub = p_vals[::2]
                    ta_sub = ta_vals[::2]
                    t_sub = t_vals[::2]
                    
                    t_q = quantize_temperature(t_sub, q=q)
                    mask = apply_sparsity_mask(t_q, coverage=cov, strategy="uniform", seed=seed)
                    
                    t_obs, p_obs = t_q[mask], p_sub[mask]
                    if len(t_obs) >= 3:
                        try:
                            fit_ols = ClassicalBaselineModel(dt_s=dt_s).fit([(t_obs, p_obs)])
                            tau_ols = float(fit_ols.tau_eff_s)
                            K_ols = float(fit_ols.r_eff / tau_ols) if tau_ols > 0 else float("nan")
                            if np.isfinite(tau_ols):
                                ols_taus_seeds.append(tau_ols)
                                ols_Ks_seeds.append(K_ols)
                        except Exception:
                            pass
                            
                    time_obs = time_sub[mask]
                    
                    pinn_model = TemperaturePINN(hidden_dim=64, tau_init=390.0, K_init=0.05)
                    res_pinn = train_pinn(
                        model=pinn_model,
                        t_obs=time_obs,
                        p_obs=p_obs,
                        ta_obs=ta_sub[mask],
                        temp_obs=t_obs,
                        t_colloc=time_sub,
                        p_colloc=p_sub,
                        ta_colloc=ta_sub,
                        epochs=200,
                        lr=3e-3,
                        seed=seed,
                    )
                    tau_pinn = float(res_pinn["tau_fitted"])
                    K_pinn = float(res_pinn["K_fitted"])
                    
                    pinn_taus_seeds.append(tau_pinn)
                    pinn_Ks_seeds.append(K_pinn)
                    
            tau_ref = 393.8
            med_pinn_tau = float(np.nanmedian(pinn_taus_seeds))
            med_pinn_K = float(np.nanmedian(pinn_Ks_seeds))
            med_ols_tau = float(np.nanmedian(ols_taus_seeds)) if ols_taus_seeds else float("nan")
            med_ols_K = float(np.nanmedian(ols_Ks_seeds)) if ols_Ks_seeds else float("nan")
            
            pinn_err_pct = float(abs(med_pinn_tau - tau_ref) / tau_ref * 100.0)
            ols_err_pct = float(abs(med_ols_tau - tau_ref) / tau_ref * 100.0) if np.isfinite(med_ols_tau) else 100.0
            
            p05, p95 = np.nanpercentile(pinn_taus_seeds, [5, 95])
            ci_width_pct = float((p95 - p05) / tau_ref * 100.0)
            
            is_reliable_10 = bool(check_reliability(pinn_err_pct, ci_width_pct, tol_err=10.0, tol_ci=20.0))
            is_reliable_5 = bool(check_reliability(pinn_err_pct, ci_width_pct, tol_err=5.0, tol_ci=10.0))
            is_reliable_20 = bool(check_reliability(pinn_err_pct, ci_width_pct, tol_err=20.0, tol_ci=30.0))
            
            grid_results.append({
                "coverage": float(cov),
                "quantization_q": float(q),
                "dt_s": float(dt_s),
                "tau_pinn_raw_s": med_pinn_tau,
                "K_pinn_raw_KW": med_pinn_K,
                "tau_ols_raw_s": med_ols_tau,
                "K_ols_raw_KW": med_ols_K,
                "pinn_err_pct": pinn_err_pct,
                "ols_err_pct": ols_err_pct,
                "ci95_width_pct": ci_width_pct,
                "is_reliable_10pct": is_reliable_10,
                "is_reliable_5pct": is_reliable_5,
                "is_reliable_20pct": is_reliable_20,
            })
            
            print(f"Cov: {cov*100:4.0f}% | q: {q:3.1f}C | PINN tau: {med_pinn_tau:5.1f}s, K: {med_pinn_K:.5f} | "
                  f"Err: {pinn_err_pct:4.1f}% | CI Width: {ci_width_pct:4.1f}% | Reliable (10%): {is_reliable_10}")
            
    summary = {
        "phase": "2G5_combined",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_grid_cells": len(grid_results),
        "results": grid_results,
    }
    
    out_file = RESULTS_DIR / "phase2g5_combined.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G5 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g5()
