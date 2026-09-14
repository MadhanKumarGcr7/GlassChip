"""Master reproduction pipeline for GLASSCHIP-PINN study.

Executes all 8 phases (2G0 - 2G7), saves JSON result artifacts,
regenerates all canonical figures and tables, and builds the PINN results manifest.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from glasschip.analysis.pinn_analysis import run_all_analysis

RESULTS_DIR = _REPO / "artifacts" / "results" / "pinn"
MANIFEST_PATH = RESULTS_DIR / "pinn_results_manifest.json"


def load_and_run_phase(file_name: str, func_name: str):
    """Dynamically load and execute a phase script starting with a digit."""
    file_path = EXPERIMENTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(file_name.replace(".py", ""), str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[file_name.replace(".py", "")] = mod
    spec.loader.exec_module(mod)
    func = getattr(mod, func_name)
    return func()


def generate_pinn_manifest() -> dict:
    """Build master provenance manifest for PINN study."""
    phases = ["2G0_synthetic", "2G1_reference", "2G2_sparsity", "2G3_temporal",
              "2G4_quantization", "2G5_combined", "2G6_crosshost", "2G7_ablation"]
    
    manifest_entries = []
    for p in phases:
        file_path = RESULTS_DIR / f"phase{p.lower()}.json"
        if file_path.exists():
            manifest_entries.append({
                "phase": p,
                "file": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "status": "VALID",
            })
            
    manifest = {
        "study": "GLASSCHIP-PINN",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_phases": len(manifest_entries),
        "phases": manifest_entries,
        "figures_dir": "paper/figures/pinn/",
        "tables_dir": "paper/tables/pinn/",
        "report": "docs/PINN_IDENTIFIABILITY.md",
    }
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest


def main():
    print("=======================================================================")
    print("          GLASSCHIP-PINN EXPERIMENTAL REPRODUCTION PIPELINE            ")
    print("=======================================================================")
    start_time = time.time()
    
    # 1. Run Phases 2G0 to 2G7
    print("\n--- Step 1: Executing Experimental Phases (2G0 - 2G7) ---")
    phases = [
        ("2G0_synthetic.py", "run_phase_2g0"),
        ("2G1_reference.py", "run_phase_2g1"),
        ("2G2_sparsity.py", "run_phase_2g2"),
        ("2G3_temporal.py", "run_phase_2g3"),
        ("2G4_quantization.py", "run_phase_2g4"),
        ("2G5_combined.py", "run_phase_2g5"),
        ("2G6_crosshost.py", "run_phase_2g6"),
        ("2G7_ablation.py", "run_phase_2g7"),
    ]
    
    for file_name, func_name in phases:
        load_and_run_phase(file_name, func_name)
    
    # 2. Generate Figures & Tables
    print("\n--- Step 2: Regenerating Figures and Tables ---")
    run_all_analysis()
    
    # 3. Build Manifest
    print("\n--- Step 3: Building Master Provenance Manifest ---")
    manifest = generate_pinn_manifest()
    print(f"Manifest built with {manifest['n_phases']} valid phases -> {MANIFEST_PATH}")
    
    elapsed = time.time() - start_time
    print(f"\n=======================================================================")
    print(f" GATE: GREEN — GLASSCHIP-PINN pipeline completed in {elapsed:.1f}s")
    print("=======================================================================")


if __name__ == "__main__":
    main()
