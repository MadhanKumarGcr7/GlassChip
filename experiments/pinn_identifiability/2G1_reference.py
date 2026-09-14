"""GLASSCHIP-PINN Phase 2G1 - Summit Reference Channel Identification."""

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
from pinn_utils import get_pilot_hosts, load_or_generate_host_trajectory

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_2g1() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G1: Summit Reference Identification ===")
    
    hosts = get_pilot_hosts()[:3]
    host_results = []
    
    taus_ols = []
    taus_pinn = []
    
    for host in hosts:
        for socket in [0, 1]:
            time_vals, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=socket)
            
            ols_model = ClassicalBaselineModel(dt_s=10.0)
            ols_fit = ols_model.fit([(t_vals, p_vals)])
            tau_ols = ols_fit.tau_eff_s
            K_ols = ols_fit.r_eff / tau_ols if tau_ols > 0 else float("nan")
            
            pinn_model = TemperaturePINN(
                hidden_dim=64,
                tau_init=tau_ols if np.isfinite(tau_ols) else 390.0,
                K_init=0.05,
            )
            
            res_pinn = train_pinn(
                model=pinn_model,
                t_obs=time_vals,
                p_obs=p_vals,
                ta_obs=ta_vals,
                temp_obs=t_vals,
                epochs=250,
                lr=3e-3,
                lam_data=1.0,
                lam_phys=0.05,
                seed=42,
            )
            
            tau_pinn = res_pinn["tau_fitted"]
            K_pinn = res_pinn["K_fitted"]
            
            taus_ols.append(tau_ols)
            taus_pinn.append(tau_pinn)
            
            rmse_ols = float(ols_model.evaluate([(t_vals, p_vals)]).rmse)
            
            host_results.append({
                "host": host,
                "socket": socket,
                "tau_ols": tau_ols,
                "K_ols": K_ols,
                "rmse_ols": rmse_ols,
                "tau_pinn": tau_pinn,
                "K_pinn": K_pinn,
            })
            
            print(f"Host {host} Socket {socket} | OLS tau: {tau_ols:.1f} s | PINN tau: {tau_pinn:.1f} s")
            
    summary = {
        "phase": "2G1_reference",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_units": len(host_results),
        "tau_ols_median": float(np.nanmedian(taus_ols)),
        "tau_pinn_median": float(np.nanmedian(taus_pinn)),
        "units": host_results,
    }
    
    out_file = RESULTS_DIR / "phase2g1_reference.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G1 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g1()
