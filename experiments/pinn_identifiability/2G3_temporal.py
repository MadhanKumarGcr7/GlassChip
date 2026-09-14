"""GLASSCHIP-PINN Phase 2G3 - Temporal Downsampling Degradation Study."""

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


def run_phase_2g3() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G3: Temporal Downsampling Study ===")
    
    dt_levels = [10.0, 20.0, 40.0, 60.0]
    hosts = get_pilot_hosts()[:3]
    
    results = []
    
    for dt_target in dt_levels:
        decimate_factor = int(dt_target // 10.0)
        
        ols_taus = []
        pinn_taus = []
        
        for host in hosts:
            time_vals, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0)
            
            time_sub = time_vals[::decimate_factor]
            p_sub = p_vals[::decimate_factor]
            ta_sub = ta_vals[::decimate_factor]
            t_sub = t_vals[::decimate_factor]
            
            ols_model = ClassicalBaselineModel(dt_s=dt_target)
            ols_fit = ols_model.fit([(t_sub, p_sub)])
            tau_ols = ols_fit.tau_eff_s
            
            pinn_model = TemperaturePINN(hidden_dim=64, tau_init=390.0, K_init=0.05)
            
            res_pinn = train_pinn(
                model=pinn_model,
                t_obs=time_sub,
                p_obs=p_sub,
                ta_obs=ta_sub,
                temp_obs=t_sub,
                epochs=800,
                lr=2e-3,
                lam_data=1.0,
                lam_phys=0.05,
                seed=42,
            )
            tau_pinn = res_pinn["tau_fitted"]
            
            if np.isfinite(tau_ols):
                ols_taus.append(tau_ols)
            pinn_taus.append(tau_pinn)
            
        median_ols = float(np.nanmedian(ols_taus))
        median_pinn = float(np.nanmedian(pinn_taus))
        
        print(f"dt = {dt_target:2.0f} s | OLS median tau: {median_ols:5.1f} s | PINN median tau: {median_pinn:5.1f} s")
        
        results.append({
            "dt_s": dt_target,
            "decimate_factor": decimate_factor,
            "tau_ols_median": median_ols,
            "tau_pinn_median": median_pinn,
        })
        
    summary = {
        "phase": "2G3_temporal",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_experiments": len(results),
        "results": results,
    }
    
    out_file = RESULTS_DIR / "phase2g3_temporal.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G3 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g3()
