# Results — Qwen2.5-14B-Instruct-AWQ on a single RTX 4090 (24 GB)

Baseline sweep, FP16 KV cache, `max-model-len=16384`, `gpu-memory-utilization=0.90`.
Source data: [`results/benchmark_results.csv`](../results/benchmark_results.csv),
visual: [`results/feasibility_heatmap.png`](../results/feasibility_heatmap.png).

## Heatmap

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **1024**  | feasible | feasible | feasible | feasible |
| **4096**  | feasible | feasible | marginal | marginal |
| **8192**  | marginal | marginal | marginal | marginal |
| **16384** | infeasible | infeasible | infeasible | infeasible |

Counts: `feasible=6, marginal=6, infeasible=4`.

## Headline numbers

| Question | Answer |
|---|---|
| Best throughput, feasible cell | **305 tok/s** at ctx=1024, conc=8 (peak VRAM 21.8 GB) |
| Largest production-safe context | **4096 tokens at conc≤2**, p95 latency 3.4 s |
| Best per-user latency | TPOT p50 = 12 ms across the entire 1k–4k row |
| VRAM ceiling reached | **22.85 GB** — saturates at every cell with ctx≥4k or conc≥4 |

## Reading the boundary

Three things the data says clearly:

1. **The wall is VRAM, not compute.** Every "marginal" cell pegs at exactly
   22,846 MB peak — that's vLLM's pre-allocated KV pool (sized to
   `max-model-len × max-num-seqs`) saturating between the 22 GB feasibility
   ceiling and the 24 GB hard limit. The card has compute to spare; KV memory
   is what runs out.

2. **Latency degrades smoothly until 8 k.** TPOT p95 stays under 22 ms through
   the entire 4 k row. By 8 k×8 it climbs to 67 ms (~15 tok/s per user) — slow
   but functional. The absolute p95 latencies for an 8 k context with 8
   concurrent users approach 19 s, which is the SLO ceiling for chat-style apps.

3. **Continuous batching works.** Throughput scales from 74 → 305 tok/s as
   concurrency goes 1 → 8 at ctx=1024 (4× scaling for 8× concurrency, the
   shortfall is prefill serialization). At 8 k the same 1 → 8 sweep gives only
   37 → 55 tok/s — KV pressure crowds out batch parallelism.

## Why the 16 k row is `http_error`, not `oom`

The four 16k cells failed instantly with HTTP 400 (~80–600 ms wall time, no
GPU work). This is a **measurement artifact, not a hardware limit**: the
runner sends `prompt_tokens ≈ 16384` plus `max_tokens=128` plus chat-template
overhead, asking for ~16,500+ tokens against a server pinned at
`max-model-len=16384`. vLLM rejects the request before prefill.

To measure 16 k honestly, raise the server cap. That's experiment #2 below.

## What this implies for deployment

Mapped onto realistic workload classes:

| Workload | Verdict |
|---|---|
| Chatbot / instruction-following, ≤4 k context, up to 8 concurrent users | **Safe.** Best-case 305 tok/s aggregate, 12 ms TPOT. |
| RAG with ~4 k context, 1–2 concurrent users | **Safe.** p95 latency under 3.4 s. |
| RAG with ~4 k context, 4–8 concurrent users | **Marginal.** Works, but no headroom for traffic spikes. Plan a second card. |
| Long-document summarization (8 k input, 1–2 users) | **Marginal.** TTFT ~3.4 s, latency ~5.3 s — usable, not snappy. |
| 16 k context anything | **Not measured cleanly here** — see experiment #2. |
| 32 B model in any flavor | **Likely infeasible** at FP16 KV — 14 B already eats 22.8 GB at 4 k+. See experiment #3. |

## Knobs that should move the boundary (untested in this sweep)

Two cheap, well-known levers — both worth one sweep each before declaring the
24 GB card "fully characterized":

- **FP8 KV cache** (`KV_CACHE_DTYPE=fp8`): roughly halves KV memory at minor
  accuracy cost. Predicted to turn the 8 k row green and possibly unlock 16 k
  at low concurrency.
- **Prefix caching** (`--enable-prefix-caching`): orthogonal to this sweep —
  helps only when prompts share a long fixed prefix (system prompt, tool
  schema). Out of scope for a *capacity* characterization but worth a footnote
  for any RAG deployment.
