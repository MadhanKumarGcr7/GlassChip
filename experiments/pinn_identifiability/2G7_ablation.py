"""GLASSCHIP-PINN Phase 2G7 - Comprehensive Ablation Study."""

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
from glasschip.models.pinn import TemperaturePINN, HybridTemperaturePINN, train_pinn
from pinn_utils import (
    get_pilot_hosts,
    load_or_generate_host_trajectory,
    apply_sparsity_mask,
    quantize_temperature,
)

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_2g7() -> Dict[str, Any]:
    print("=== Running GLASSCHIP-PINN Phase 2G7: Ablation Study ===")
    
    host = get_pilot_hosts()[0]
    time_vals, p_vals, ta_vals, t_vals = load_or_generate_host_trajectory(host, socket=0)
    
    ols_ref = ClassicalBaselineModel(dt_s=10.0).fit([(t_vals, p_vals)])
    tau_ref = float(ols_ref.tau_eff_s)
    
    ablations = []
    
    pinn_no_phys = TemperaturePINN(hidden_dim=64, tau_init=350.0)
    res_no_phys = train_pinn(pinn_no_phys, time_vals, p_vals, ta_vals, t_vals, epochs=250, lam_phys=0.0)
    
    pinn_with_phys = TemperaturePINN(hidden_dim=64, tau_init=350.0)
    res_with_phys = train_pinn(pinn_with_phys, time_vals, p_vals, ta_vals, t_vals, epochs=250, lam_phys=0.1)
    
    ablations.append({
        "ablation_id": "A1_loss_component",
        "name": "Physics Loss Contribution",
        "tau_no_phys": float(res_no_phys["tau_fitted"]),
        "tau_with_phys": float(res_with_phys["tau_fitted"]),
        "err_no_phys_pct": float(abs(res_no_phys["tau_fitted"] - tau_ref) / tau_ref * 100.0),
        "err_with_phys_pct": float(abs(res_with_phys["tau_fitted"] - tau_ref) / tau_ref * 100.0),
    })
    
    hybrid_model = HybridTemperaturePINN(hidden_dim=64, tau_init=350.0, max_discrepancy=0.05)
    res_hybrid = train_pinn(hybrid_model, time_vals, p_vals, ta_vals, t_vals, epochs=250, lam_phys=0.1)
    
    ablations.append({
        "ablation_id": "A2_model_formulation",
        "name": "Strict vs Hybrid PINN",
        "tau_strict": float(res_with_phys["tau_fitted"]),
        "tau_hybrid": float(res_hybrid["tau_fitted"]),
        "err_strict_pct": float(abs(res_with_phys["tau_fitted"] - tau_ref) / tau_ref * 100.0),
        "err_hybrid_pct": float(abs(res_hybrid["tau_fitted"] - tau_ref) / tau_ref * 100.0),
    })
    
    t_q1 = quantize_temperature(t_vals, q=1.0)
    pinn_quant = TemperaturePINN(hidden_dim=64, tau_init=350.0)
    res_quant = train_pinn(pinn_quant, time_vals, p_vals, ta_vals, t_q1, epochs=250, lam_phys=0.1)
    
    ablations.append({
        "ablation_id": "A3_quantization_impact",
        "name": "Quantization Impact (1 degC)",
        "tau_unquantized": float(res_with_phys["tau_fitted"]),
        "tau_quantized": float(res_quant["tau_fitted"]),
        "err_unquant_pct": float(abs(res_with_phys["tau_fitted"] - tau_ref) / tau_ref * 100.0),
        "err_quant_pct": float(abs(res_quant["tau_fitted"] - tau_ref) / tau_ref * 100.0),
    })
    
    mask_rand = apply_sparsity_mask(t_vals, coverage=0.25, strategy="uniform")
    mask_block = apply_sparsity_mask(t_vals, coverage=0.25, strategy="block")
    
    pinn_rand = TemperaturePINN(hidden_dim=64, tau_init=350.0)
    res_rand = train_pinn(pinn_rand, time_vals[mask_rand], p_vals[mask_rand], ta_vals[mask_rand], t_vals[mask_rand], epochs=250, lam_phys=0.1)
    
    pinn_block = TemperaturePINN(hidden_dim=64, tau_init=350.0)
    res_block = train_pinn(pinn_block, time_vals[mask_block], p_vals[mask_block], ta_vals[mask_block], t_vals[mask_block], epochs=250, lam_phys=0.1)
    
    ablations.append({
        "ablation_id": "A4_dropout_pattern",
        "name": "Random vs Block Dropout",
        "tau_random": float(res_rand["tau_fitted"]),
        "tau_block": float(res_block["tau_fitted"]),
        "err_random_pct": float(abs(res_rand["tau_fitted"] - tau_ref) / tau_ref * 100.0),
        "err_block_pct": float(abs(res_block["tau_fitted"] - tau_ref) / tau_ref * 100.0),
    })
    
    seed_taus = []
    for s in [42, 43, 44, 45, 46]:
        pm = TemperaturePINN(hidden_dim=64, tau_init=350.0)
        r = train_pinn(pm, time_vals, p_vals, ta_vals, t_vals, epochs=250, seed=s)
        seed_taus.append(float(r["tau_fitted"]))
        
    cv_seeds = float(np.std(seed_taus) / np.mean(seed_taus) * 100.0)
    ablations.append({
        "ablation_id": "A5_seed_variance",
        "name": "Multi-seed Variance (5 seeds)",
        "seed_taus": seed_taus,
        "mean_tau": float(np.mean(seed_taus)),
        "std_tau": float(np.std(seed_taus)),
        "cv_pct": cv_seeds,
    })
    
    for a in ablations:
        print(f"Ablation {a['ablation_id']} ({a['name']}) complete.")
        
    summary = {
        "phase": "2G7_ablation",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_ablations": len(ablations),
        "ablations": ablations,
    }
    
    out_file = RESULTS_DIR / "phase2g7_ablation.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Phase 2G7 complete. Output saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_phase_2g7()
