# Table 2. Telemetry Degradation Matrix and Parameter Recovery Performance

| Condition | Sensor Coverage (%) | Quantization $q$ (°C) | Sampling $\Delta t$ (s) | Raw PINN $\tau$ (s) | Raw PINN $K$ (K/W) | OLS $\tau$ Error (%) | PINN $\tau$ Error (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Reference (F0) | 100% | 0.0 | 10s | 280.3s | 0.03810 | 0.0% | 28.0% |
| Quantized Only (F1) | 100% | 1.0 | 10s | 61.5s | 0.00940 | 0.0% | 28.0% |
| Downsampled Only (F2) | 100% | 0.0 | 20s | 61.4s | 0.00940 | 0.0% | 28.0% |
| Moderate Sparsity | 50% | 0.0 | 10s | 280.3s | 0.03810 | 57.5% | 28.1% |
| High Sparsity | 25% | 0.0 | 10s | 280.3s | 0.03810 | 80.4% | 28.1% |
| Severe Sparsity | 10% | 0.0 | 10s | 280.3s | 0.03810 | 89.5% | 28.1% |
| Combined Stress (F4) | 25% | 1.0 | 20s | 433.7s | 0.05200 | 100.0% | 10.1% |
