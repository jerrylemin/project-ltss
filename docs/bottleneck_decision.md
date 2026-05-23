# Bottleneck Decision

Last updated: 2026-05-23T04:13:29.590602+00:00

The dominant bottleneck is the PageRank iteration SpMV step: each iteration scans the incoming CSR structure and gathers `rank[src] / out_degree[src]` for every edge.

This makes the project a good fit for partial GPU acceleration because the graph traversal and vector update are repeated many times while configuration, loading, validation, and reporting remain on CPU.

## Profile Summary

- Graph: synthetic_small
- Nodes: 1000
- Edges: 5000
- Elapsed seconds: 0.14584500000000844
- Iterations: 16

## Top Functions by Cumulative Time

- `pagerank_cpu.py:91:run_pagerank_cpu`: cumtime=0.145860s, calls=1
- `pagerank_cpu.py:9:_pagerank_numpy_loop`: cumtime=0.145837s, calls=1
- `fromnumeric.py:2177:sum`: cumtime=0.082771s, calls=15888
- `fromnumeric.py:71:_wrapreduction`: cumtime=0.061356s, calls=15888
- `~:0:<method 'reduce' of 'numpy.ufunc' objects>`: cumtime=0.028001s, calls=15937
- `fromnumeric.py:72:<dictcomp>`: cumtime=0.010029s, calls=15888
- `~:0:<built-in method builtins.isinstance>`: cumtime=0.002869s, calls=15888
- `fromnumeric.py:2172:_sum_dispatcher`: cumtime=0.002638s, calls=15888
- `~:0:<method 'items' of 'dict' objects>`: cumtime=0.001967s, calls=15888
- `~:0:<method 'sum' of 'numpy.ndarray' objects>`: cumtime=0.000197s, calls=49
