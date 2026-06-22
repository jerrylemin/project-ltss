# Bottleneck Decision

Last updated: 2026-06-13

The dominant CPU bottleneck is the PageRank SpMV step: each iteration scans incoming CSR edges and gathers `rank[src] / out_degree[src]` for every destination.

This is measured on a real SNAP graph, not inferred from synthetic smoke data.

## Profile Summary

- Graph: `amazon0601`
- Nodes: `403,394`
- Edges: `3,387,388`
- Iterations: `55`
- Full convergence time: `2.821893s`
- Total measured iteration compute time: `2.819990s`
- SpMV time: `2.573286s`
- Damping time: `0.084791s`
- L1 convergence time: `0.099163s`
- SpMV percentage of iteration time: `91.37%`

## Decision

SpMV remains the correct GPU acceleration target. Damping, dangling mass, normalization, and L1 convergence are smaller but still worth fusing or reducing on device, which is why V2 fuses SpMV+damping+L1 and V3 uses warp-level shuffle reduction.
