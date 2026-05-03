# Results — Qwen2.5-32B-Instruct-AWQ + FP8 KV cache (RTX 4090, 24 GB)

This is the "can we fit the next model size up?" experiment. 32B AWQ weights
are ~17 GB resident vs 14B's ~8 GB, leaving roughly 5–6 GB on a 24 GB card
for KV cache + activations.

Launch defaults (from [`scripts/start_vllm_qwen32b_awq.sh`](../scripts/start_vllm_qwen32b_awq.sh)):
`max-model-len=8192`, `max-num-seqs=4`, `kv-cache-dtype=fp8`,
`gpu-memory-utilization=0.92`. The sweep grid was correspondingly narrower
(no ctx=16384, no conc=8 — they have no chance of fitting).

Source data: [`results/benchmark_results_32b.csv`](../results/benchmark_results_32b.csv),
visual: [`results/32b/feasibility_heatmap.png`](../results/32b/feasibility_heatmap.png).

## Heatmap

| ctx \ conc | 1 | 2 | 4 |
|---|---|---|---|
| **1024** | feasible | marginal | marginal |
| **2048** | marginal | marginal | marginal |
| **4096** | infeasible | infeasible | infeasible |
| **8192** | infeasible | infeasible | infeasible |

Counts: `feasible=1, marginal=5, infeasible=6`.

## The single feasible cell

| metric | value |
|---|---|
| ctx, conc | 1024, 1 |
| TTFT p95 | 464 ms |
| TPOT p95 | 23 ms (~43 tok/s per user) |
| Latency p95 | 3.4 s |
| Throughput | 37.8 tok/s aggregate |
| Peak VRAM | 22.4 GB / 24 GB |

Even the "feasible" corner has ~600 MB of headroom. Compare to 14B baseline at
the same cell: 74 tok/s and 12 ms TPOT, with 4 GB VRAM headroom. **32B halves
your per-user throughput and uses 80% more VRAM** for the same workload.

## Why everything else fails

Two failure modes mixed together:

1. **VRAM saturation** at conc≥2 even at ctx=1024 — peak VRAM goes 22.4 → 22.7
   → 23.1 GB. Anything past 22 GB trips the `feasible_max_mb` ceiling. The
   1024,2 and 1024,4 cells *work* but are labeled marginal because they're
   one traffic spike away from OOM.
2. **`http_error` at ctx≥4096** — the 4 k row fails instantly (<60 ms wall),
   meaning the request was rejected, not OOM'd at runtime. With `max-num-seqs=4`
   and a per-request KV budget that vLLM computes conservatively under
   memory pressure, requests for 4 k+ context get rejected. The 8 k row fails
   for the obvious reason (`max-model-len=8192` minus chat overhead minus
   `max_tokens=128` leaves no room).

## Conclusions

For the project's headline question — "what can a single 24 GB GPU serve?" —
the answer for Qwen2.5-32B-AWQ is sharp:

- **It boots and runs**, but only at ctx=1024 with concurrency=1.
- That configuration is roughly half the throughput of 14B on the same card,
  with no headroom for traffic.
- **In practical terms: don't deploy 32B on a single 24 GB card.** It is a
  capability demonstration, not a serving target.

If 32B is required for quality reasons, the cheapest fixes (in order):

1. **Multi-GPU.** Two 24 GB cards via tensor-parallel turn 32B from "barely
   fits" into "fits comfortably with room for ctx≥4 k". Real production answer.
2. **Higher-VRAM single card** (A100 40 GB / L40S 48 GB / H100 80 GB).
   Solves it but costs ~4–10× per hour.
3. **Drop to 14B-AWQ.** If the quality delta is acceptable for your task,
   the 14B numbers in [`results_qwen14b.md`](results_qwen14b.md) are
   dramatically better at every workload size.
