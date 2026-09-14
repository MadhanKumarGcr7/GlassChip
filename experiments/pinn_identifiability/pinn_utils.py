"""Shared utilities for PINN identifiability experiment scripts (Phases 2G0 - 2G7)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

PHASE2B_FILE = _REPO / "artifacts" / "results" / "phase2b_ablation.json"


def get_pilot_hosts() -> List[str]:
    """Retrieve 10 pilot hosts from committed Phase 2B results or manifests."""
    if PHASE2B_FILE.exists():
        data = json.loads(PHASE2B_FILE.read_text())
        return data.get("hosts", ["g04n06", "g14n16", "g15n08", "g20n18", "g25n18",
                                   "g26n10", "h25n02", "h25n12", "h31n04", "h36n14"])
    return ["g04n06", "g14n16", "g15n08", "g20n18", "g25n18", "g26n10", "h25n02", "h25n12", "h31n04", "h36n14"]


def load_or_generate_host_trajectory(
    host_name: str, socket: int = 0, dt: float = 10.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load real parquet telemetry if present, or generate host-realistic trajectory."""
    cleaned_dir = Path(os.environ.get("GLASSCHIP_SUMMIT_DERIVED", _REPO / "data" / "summit" / "derived" / "cleaned"))
    parquet_path = cleaned_dir / f"{host_name}.parquet"
    
    if parquet_path.exists():
        try:
            import polars as pl
            df = pl.read_parquet(parquet_path)
            t_col = f"p{socket}_core_temp_mean"
            p_col = f"p{socket}_power"
            if t_col in df.columns and p_col in df.columns:
                sub = df.select(["timestamp", t_col, p_col]).drop_nulls()
                t_vals = sub[t_col].to_numpy().astype(float)
                p_vals = sub[p_col].to_numpy().astype(float)
                time_vals = np.arange(len(t_vals)) * dt
                ta_vals = np.full_like(t_vals, 22.0)
                return time_vals[:1000], p_vals[:1000], ta_vals[:1000], t_vals[:1000]
        except Exception:
            pass

    rng = np.random.default_rng(seed + hash(host_name) % 10000)
    tau_ref = float(rng.uniform(320.0, 480.0))
    K_ref = float(rng.uniform(0.04, 0.08))
    
    N = 600
    time_vals = np.arange(N) * dt
    
    p_base = rng.choice([40.0, 180.0, 300.0, 80.0], size=N // 50)
    p_vals = np.repeat(p_base, 50)[:N] + rng.normal(0.0, 4.0, size=N)
    p_vals = np.clip(p_vals, 30.0, 350.0)
    
    t_vals = np.zeros(N)
    t_vals[0] = 25.0
    ta_vals = np.full(N, 22.0)
    for i in range(1, N):
        dTdt = -(1.0 / tau_ref) * (t_vals[i-1] - ta_vals[i-1]) + K_ref * p_vals[i-1]
        t_vals[i] = t_vals[i-1] + dt * dTdt
        
    return time_vals, p_vals, ta_vals, t_vals


def apply_sparsity_mask(
    t_vals: np.ndarray,
    coverage: float,
    strategy: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """Return boolean mask for observed points based on coverage (0.0 to 1.0)."""
    N = len(t_vals)
    k = max(2, int(N * coverage))
    mask = np.zeros(N, dtype=bool)
    
    if coverage >= 1.0:
        return np.ones(N, dtype=bool)
        
    rng = np.random.default_rng(seed)
    
    if strategy == "uniform":
        idx = rng.choice(N, size=k, replace=False)
        mask[idx] = True
        mask[0] = True
    elif strategy == "block":
        start_gap = rng.integers(N // 4, N // 2)
        gap_len = N - k
        mask[:] = True
        mask[start_gap : min(N, start_gap + gap_len)] = False
        mask[0] = True
    elif strategy == "periodic":
        step = max(1, int(1.0 / coverage))
        mask[::step] = True
        mask[0] = True
        
    return mask


def quantize_temperature(T: np.ndarray, q: float) -> np.ndarray:
    """Quantize scalar temperature trace to step size q degC."""
    if q <= 0.0:
        return T.copy()
    return np.round(T / q) * q
