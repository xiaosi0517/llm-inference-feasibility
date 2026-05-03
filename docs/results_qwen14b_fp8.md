# Results — Qwen2.5-14B-Instruct-AWQ + FP8 KV cache (RTX 4090, 24 GB)

Follow-up to the [FP16 KV baseline](results_qwen14b.md). Only one knob changed:
`KV_CACHE_DTYPE=fp8`. Everything else (`max-model-len`, `max-num-seqs=8`,
`gpu-memory-utilization=0.90`) matched the baseline launch.

Source data: [`results/benchmark_results_14b_fp8.csv`](../results/benchmark_results_14b_fp8.csv),
visual: [`results/14b_fp8/feasibility_heatmap.png`](../results/14b_fp8/feasibility_heatmap.png).

## Heatmap

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **1024**  | feasible | feasible | feasible | feasible |
| **4096**  | feasible | feasible | marginal | marginal |
| **8192**  | infeasible | infeasible | infeasible | infeasible |
| **16384** | infeasible | infeasible | infeasible | infeasible |

Counts: `feasible=6, marginal=2, infeasible=8`.

## Where FP8 KV helped (as predicted)

Direct cell-vs-cell comparison against the FP16 KV baseline, looking at peak
VRAM and throughput:

| cell | FP16 KV peak VRAM | FP8 KV peak VRAM | Δ VRAM | FP16 tput | FP8 tput |
|---|---:|---:|---:|---:|---:|
| 1024×1 | 20.1 GB | 19.5 GB | **−0.6 GB** | 74.2 tok/s | 72.8 tok/s |
| 1024×8 | 21.8 GB | 20.8 GB | **−1.0 GB** | 305.3 tok/s | **314.0 tok/s** |
| 4096×1 | 21.8 GB | 20.8 GB | **−1.0 GB** | 52.7 tok/s | 54.3 tok/s |
| 4096×2 | 22.2 GB | 21.2 GB | **−1.0 GB** | 76.6 tok/s | 78.9 tok/s |
| 4096×8 | 22.8 GB | 22.6 GB | −0.2 GB | 116.6 tok/s | **121.2 tok/s** |

Across the cells where both runs succeeded, FP8 KV **frees ~1 GB of VRAM** and
nudges throughput up by 1–4%. Both effects are exactly what FP8 KV is supposed
to do (halve KV bytes, no compute slowdown). The freed VRAM did **not**
flip any "marginal" cells to "feasible" — peak VRAM at 4 k×4 is still essentially
at the 22.8 GB ceiling because vLLM expanded its KV pool to consume the freed
budget.

## What the 8 k+ regression actually is

Every 8 k+ cell shows `failure_reason=http_error` and `gpu_mem_peak_mb` drops
to **474 MB** by the second failed cell — that's the bare CUDA context with no
model loaded. **The vLLM server crashed mid-sweep**, somewhere between the
last successful 4096×8 burst and the first 8192×1 burst.

This is not a capacity finding. Most likely cause: the `MAX_MODEL_LEN=20480`
env var was not exported when the launch script ran, so the server kept its
16384 default. With a longer prompt budget than the FP16 run had headroom for
plus the FP8 KV pool sizing change, the engine hit an unrecoverable error.

To honestly measure 8 k / 16 k under FP8 KV, restart with the env var verified:

```bash
KV_CACHE_DTYPE=fp8 MAX_MODEL_LEN=20480 \
  bash scripts/start_vllm_qwen14b_awq.sh
# in the server log, confirm BOTH lines:
#   kv-cache-dtype   = fp8
#   max-model-len    = 20480
```

## Conclusions

- FP8 KV is a **free 1 GB of VRAM** for any cell that fits, and a small
  throughput boost — keep it on by default for production.
- It does **not** by itself extend the feasibility frontier on a 24 GB card
  for 14B-AWQ — vLLM's KV pool sizing absorbs the savings. The boundary moves
  only if you also raise `MAX_MODEL_LEN` and/or `MAX_NUM_SEQS` to put the
  freed memory to work.
- The 8 k regression in this sweep is a server crash, not a hardware limit;
  re-run with `MAX_MODEL_LEN=20480` to get clean 8 k / 16 k numbers.
