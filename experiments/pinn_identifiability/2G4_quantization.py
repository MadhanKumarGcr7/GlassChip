"""GLASSCHIP-PINN Phase 2G4 - Temperature Quantization Degradation Study."""

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
from pinn_utils import get_pilot_hosts, load_or_generate_host_trajectory, quantize_temperature

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_2g4() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G4: Quantization Study ===")
    
    quant_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    hosts = get_pilot_hosts()[:3]
    
    results = []
    
    for q in quant_levels:
        ols_taus = []
        pinn_taus = []
        
        for host in hosts:
            time_vals, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0)
            
            t_q = quantize_temperature(t_vals, q=q)
            
            ols_model = ClassicalBaselineModel(dt_s=10.0)
            ols_fit = ols_model.fit([(t_q, p_vals)])
            tau_ols = ols_fit.tau_eff_s
            
            pinn_model = TemperaturePINN(hidden_dim=64, tau_init=390.0, K_init=0.05)
            
            res_pinn = train_pinn(
                model=pinn_model,
                t_obs=time_vals,
                p_obs=p_vals,
                ta_obs=ta_vals,
                temp_obs=t_q,
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
        
        print(f"q = {q:4.1f} degC | OLS median tau: {median_ols:5.1f} s | PINN median tau: {median_pinn:5.1f} s")
        
        results.append({
            "quantization_q": q,
            "tau_ols_median": median_ols,
            "tau_pinn_median": median_pinn,
        })
        
    summary = {
        "phase": "2G4_quantization",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_experiments": len(results),
        "results": results,
    }
    
    out_file = RESULTS_DIR / "phase2g4_quantization.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G4 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g4()
