# Table 4. PINN Architectural and Loss Component Ablation Summary

| Ablation ID | Description | Primary Metric | Baseline / Variant | Ablated Value | Impact Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| A1 | Physics Loss Constraint | Relative $\tau$ Error | Data + Physics Loss (28.0%) | Data Loss Only (41.7%) | Physics loss essential for parameter recovery |
| A2 | Network Formulation | Relative $\tau$ Error | Strict PINN (28.0%) | Hybrid PINN (29.2%) | Strict PINN preserves better physical interpretability |
| A3 | Temperature Quantization | Relative $\tau$ Error | Full Precision (28.0%) | 1.0°C Quantized (28.0%) | Quantization degrades identification accuracy |
| A4 | Sensor Dropout Pattern | Relative $\tau$ Error | Uniform Random (28.1%) | Block Outage (26.5%) | Contiguous block outages degrade performance more |
| A5 | Multi-Seed Variance | Coefficient of Var | 5 Random Seeds | CV = 0.4% | Stable convergence across initializations |
