# Decision matrix

| ctx | conc | feas | TTFT p95 (ms) | TPOT p95 (ms) | latency p95 (ms) | tput (tok/s) | peak VRAM (MB) | reason |
|---:|---:|:---|---:|---:|---:|---:|---:|:---|
| 1024 | 1 | feasible | 464.0 | 23.0 | 3387.4 | 37.8 | 22386 | ok |
| 1024 | 2 | marginal | 901.1 | 23.5 | 3881.7 | 66.1 | 22722 | ok |
| 1024 | 4 | marginal | 1784.9 | 26.9 | 4806.6 | 106.7 | 23058 | ok |
| 2048 | 1 | marginal | 889.4 | 23.1 | 3824.1 | 33.5 | 23058 | ok |
| 2048 | 2 | marginal | 1764.3 | 28.7 | 4751.1 | 54.0 | 23058 | ok |
| 2048 | 4 | marginal | 3517.2 | 43.3 | 6623.9 | 77.5 | 23058 | ok |
| 4096 | 1 | infeasible | — | — | — | — | 23058 | http_error |
| 4096 | 2 | infeasible | — | — | — | — | 23058 | http_error |
| 4096 | 4 | infeasible | — | — | — | — | 23058 | http_error |
| 8192 | 1 | infeasible | — | — | — | — | 23058 | http_error |
| 8192 | 2 | infeasible | — | — | — | — | 23058 | http_error |
| 8192 | 4 | infeasible | — | — | — | — | 23058 | http_error |
