# Decision matrix

| ctx | conc | feas | TTFT p95 (ms) | TPOT p95 (ms) | latency p95 (ms) | tput (tok/s) | peak VRAM (MB) | reason |
|---:|---:|:---|---:|---:|---:|---:|---:|:---|
| 1024 | 1 | feasible | 207.9 | 12.3 | 1771.5 | 72.8 | 19466 | ok |
| 1024 | 2 | feasible | 401.7 | 13.6 | 1974.6 | 130.2 | 19686 | ok |
| 1024 | 4 | feasible | 789.8 | 15.9 | 2392.9 | 215.6 | 20022 | ok |
| 1024 | 8 | feasible | 1604.1 | 18.3 | 3274.6 | 314.0 | 20806 | ok |
| 4096 | 1 | feasible | 814.8 | 12.2 | 2364.7 | 54.3 | 20806 | ok |
| 4096 | 2 | feasible | 1621.4 | 13.0 | 3258.6 | 78.9 | 21242 | ok |
| 4096 | 4 | marginal | 3209.5 | 13.9 | 4969.1 | 103.1 | 22940 | ok |
| 4096 | 8 | marginal | 6356.6 | 23.2 | 8461.9 | 121.2 | 22550 | ok |
| 8192 | 1 | infeasible | — | — | — | — | 22550 | http_error |
| 8192 | 2 | infeasible | — | — | — | — | 22550 | http_error |
| 8192 | 4 | infeasible | — | — | — | — | 474 | http_error |
| 8192 | 8 | infeasible | — | — | — | — | 474 | http_error |
| 16384 | 1 | infeasible | — | — | — | — | 474 | http_error |
| 16384 | 2 | infeasible | — | — | — | — | 474 | http_error |
| 16384 | 4 | infeasible | — | — | — | — | 474 | http_error |
| 16384 | 8 | infeasible | — | — | — | — | 474 | http_error |
