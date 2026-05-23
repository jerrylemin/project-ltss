# Optimization Plan

| Version | Strategy | Target Speedup | Status |
|---------|----------|----------------|--------|
| CPU Reference | NumPy CSR loop + SciPy verify | baseline | implemented |
| GPU V1 | CSR SpMV + damping kernels | 5-15x | implemented, optional CUDA |
| GPU V2 | Fused SpMV+damping+L1 reduction | 15-40x | implemented, optional CUDA |
| GPU V3 | Push vs Pull comparison, warp reduction | 30-60x | implemented, optional CUDA |

CPU remains the correctness oracle. GPU rows are skipped cleanly if CUDA is unavailable.
