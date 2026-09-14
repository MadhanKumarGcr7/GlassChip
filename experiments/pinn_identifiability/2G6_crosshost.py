"""GLASSCHIP-PINN Phase 2G6 - Cross-Host Generalization Study."""

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


def run_phase_2g6() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G6: Cross-Host Generalization ===")
    
    all_hosts = get_pilot_hosts()
    train_hosts = all_hosts[:3]
    test_hosts = all_hosts[3:6]
    
    train_pinn_taus = []
    for host in train_hosts:
        t_vals_time, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0)
        ols_fit = ClassicalBaselineModel(dt_s=10.0).fit([(t_vals, p_vals)])
        tau_ols = ols_fit.tau_eff_s
        
        pinn_model = TemperaturePINN(hidden_dim=64, tau_init=tau_ols if np.isfinite(tau_ols) else 390.0)
        res_pinn = train_pinn(
            model=pinn_model,
            t_obs=t_vals_time,
            p_obs=p_vals,
            ta_obs=ta_vals,
            temp_obs=t_vals,
            epochs=250,
            lr=2e-3,
            seed=42,
        )
        train_pinn_taus.append(res_pinn["tau_fitted"])
        
    tau_pinn_global = float(np.median(train_pinn_taus))
    print(f"Learned Global PINN tau from training hosts: {tau_pinn_global:.1f} s")
    
    heldout_results = []
    for host in test_hosts:
        t_vals_time, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0)
        
        ols_fit = ClassicalBaselineModel(dt_s=10.0).fit([(t_vals, p_vals)])
        tau_host_ref = float(ols_fit.tau_eff_s)
        
        pinn_model_test = TemperaturePINN(hidden_dim=64, tau_init=tau_pinn_global)
        res_test = train_pinn(
            model=pinn_model_test,
            t_obs=t_vals_time,
            p_obs=p_vals,
            ta_obs=ta_vals,
            temp_obs=t_vals,
            epochs=250,
            lr=2e-3,
            seed=42,
        )
        tau_host_pinn = float(res_test["tau_fitted"])
        
        err_heldout = float(abs(tau_host_pinn - tau_host_ref) / tau_host_ref * 100.0) if np.isfinite(tau_host_ref) else float("nan")
        
        heldout_results.append({
            "host": host,
            "tau_ref_ols": tau_host_ref,
            "tau_pinn": tau_host_pinn,
            "heldout_tau_err_pct": err_heldout,
        })
        print(f"Held-out Host {host} | Reference tau: {tau_host_ref:.1f}s | PINN tau: {tau_host_pinn:.1f}s | Err: {err_heldout:.1f}%")
        
    summary = {
        "phase": "2G6_crosshost",
        "generated": datetime.now(timezone.utc).isoformat(),
        "train_hosts": train_hosts,
        "test_hosts": test_hosts,
        "tau_pinn_global_train": tau_pinn_global,
        "heldout_units": heldout_results,
    }
    
    out_file = RESULTS_DIR / "phase2g6_crosshost.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G6 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g6()
