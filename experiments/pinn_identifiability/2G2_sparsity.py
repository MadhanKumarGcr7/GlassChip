"""GLASSCHIP-PINN Phase 2G2 - Sensor Sparsity Degradation Study.

Evaluates thermal identification and prediction under controlled temperature sensor sparsity:
    Coverage levels: 100%, 75%, 50%, 25%, 10%, 5%
    Masking strategies: Uniform random, Contiguous block, Periodic
"""

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
from pinn_utils import get_pilot_hosts, load_or_generate_host_trajectory, apply_sparsity_mask

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_2g2() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G2: Sensor Sparsity Study ===")
    
    coverages = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]
    strategies = ["uniform", "block", "periodic"]
    hosts = get_pilot_hosts()[:2]
    
    results = []
    
    for cov in coverages:
        for strat in strategies:
            pinn_taus = []
            pinn_Ks = []
            ols_taus = []
            ols_Ks = []
            
            pinn_errors = []
            ols_errors = []
            
            for host in hosts:
                time_vals, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0)
                mask = apply_sparsity_mask(t_vals, coverage=cov, strategy=strat, seed=42)
                
                ols_ref = ClassicalBaselineModel(dt_s=10.0).fit([(t_vals, p_vals)])
                tau_ref = ols_ref.tau_eff_s
                K_ref = ols_ref.r_eff / tau_ref if tau_ref > 0 else 0.05
                
                t_obs, p_obs = t_vals[mask], p_vals[mask]
                if len(t_obs) >= 3:
                    try:
                        ols_sparse = ClassicalBaselineModel(dt_s=10.0).fit([(t_obs, p_obs)])
                        tau_ols = float(ols_sparse.tau_eff_s)
                        K_ols = float(ols_sparse.r_eff / tau_ols) if tau_ols > 0 else float("nan")
                    except Exception:
                        tau_ols, K_ols = float("nan"), float("nan")
                else:
                    tau_ols, K_ols = float("nan"), float("nan")
                    
                time_obs = time_vals[mask]
                
                pinn_model = TemperaturePINN(hidden_dim=64, tau_init=350.0, K_init=0.05)
                res_pinn = train_pinn(
                    model=pinn_model,
                    t_obs=time_obs,
                    p_obs=p_obs,
                    ta_obs=ta_vals[mask],
                    temp_obs=t_obs,
                    t_colloc=time_vals,
                    p_colloc=p_vals,
                    ta_colloc=ta_vals,
                    epochs=250,
                    lr=3e-3,
                    lam_data=1.0,
                    lam_phys=0.1,
                    seed=42,
                )
                tau_pinn = float(res_pinn["tau_fitted"])
                K_pinn = float(res_pinn["K_fitted"])
                
                err_pinn = float(abs(tau_pinn - tau_ref) / tau_ref * 100.0) if np.isfinite(tau_ref) else float("nan")
                err_ols = float(abs(tau_ols - tau_ref) / tau_ref * 100.0) if np.isfinite(tau_ols) and np.isfinite(tau_ref) else float("nan")
                
                pinn_taus.append(tau_pinn)
                pinn_Ks.append(K_pinn)
                if np.isfinite(tau_ols):
                    ols_taus.append(tau_ols)
                    ols_Ks.append(K_ols)
                    
                pinn_errors.append(err_pinn)
                if np.isfinite(err_ols):
                    ols_errors.append(err_ols)
                    
            median_pinn_tau = float(np.nanmedian(pinn_taus))
            median_pinn_K = float(np.nanmedian(pinn_Ks))
            median_ols_tau = float(np.nanmedian(ols_taus)) if ols_taus else float("nan")
            median_ols_K = float(np.nanmedian(ols_Ks)) if ols_Ks else float("nan")
            
            mean_pinn_err = float(np.nanmedian(pinn_errors))
            mean_ols_err = float(np.nanmedian(ols_errors)) if ols_errors else 100.0
            
            print(f"Coverage: {cov*100:5.1f}% | Strategy: {strat:8s} | "
                  f"PINN tau: {median_pinn_tau:5.1f}s, K: {median_pinn_K:.5f} | "
                  f"OLS tau: {median_ols_tau:5.1f}s | PINN Err: {mean_pinn_err:5.1f}%")
            
            results.append({
                "coverage": float(cov),
                "strategy": strat,
                "tau_pinn_raw_s": median_pinn_tau,
                "K_pinn_raw_KW": median_pinn_K,
                "tau_ols_raw_s": median_ols_tau,
                "K_ols_raw_KW": median_ols_K,
                "tau_err_pinn_median_pct": mean_pinn_err,
                "tau_err_ols_median_pct": mean_ols_err,
            })
            
    summary = {
        "phase": "2G2_sparsity",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_experiments": len(results),
        "results": results,
    }
    
    out_file = RESULTS_DIR / "phase2g2_sparsity.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G2 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g2()
