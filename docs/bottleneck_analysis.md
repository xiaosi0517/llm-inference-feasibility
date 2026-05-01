# Bottleneck analysis

Two phases of LLM inference dominate the cost picture, and they have very
different bottlenecks. Reading the heatmap is much easier once you keep them
separate.

## Prefill vs. decode

| Phase    | What it does                          | Typical bottleneck       | Sensitive to            |
|----------|---------------------------------------|--------------------------|-------------------------|
| Prefill  | Process the entire input prompt once  | Compute-bound (FLOPs)    | context length          |
| Decode   | Generate one token at a time          | Memory-bandwidth-bound   | concurrency, KV size    |

A request's TTFT is dominated by **prefill**. TPOT is dominated by **decode**.
This is why `(ctx=16k, conc=1)` and `(ctx=1k, conc=8)` look completely different
on the heatmap even when total VRAM use is similar.

## Long context: why TTFT explodes

Prefill cost is roughly `O(L^2)` for attention plus `O(L · d_model^2)` for the
MLP, where `L` is input tokens. Doubling context roughly quadruples attention
work. On a 14B model at 16k context, prefill on an Ampere 24 GB card typically
runs **0.4–1.2 s** before the first token comes out. That's TTFT before any
queueing — in a multi-tenant deployment with concurrency, prefill *also*
serializes (vLLM batches prefill across requests but still needs the FLOPs).

**Symptom on the heatmap:** TTFT p95 grows roughly with `ctx^1.5` to `ctx^2`
along a column. Concurrency makes it worse but only linearly.

**Mitigations:**
- Reduce input tokens (RAG retrieval pruning, prompt summarization).
- Enable prefix caching if your prompts share a long fixed prefix
  (vLLM `--enable-prefix-caching`). One cached prefill amortizes across
  many requests.
- Smaller model (drops d_model and num_layers, both quadratic terms).

## High concurrency: why decode and KV memory explode

KV cache scales as

```
KV ≈ 2 · num_layers · num_kv_heads · head_dim · context_len · concurrency · bytes_per_elem
```

Both `context_len` and `concurrency` are linear multipliers. At fp16 KV, each
extra concurrent request at 16k context on Qwen2.5-14B costs roughly
~1.0 GB of KV. Eight such requests is 8 GB on top of the ~9 GB of weights.

Decode is memory-bandwidth-bound because each generated token requires reading
the entire KV cache for attention. Higher concurrency means a larger KV cache
to traverse on every step → TPOT goes up roughly linearly with concurrency
once the GPU's HBM bandwidth saturates.

**Symptom on the heatmap:** TPOT p95 grows along a row (concurrency axis).
Throughput-per-request degrades, but *aggregate* throughput keeps rising
until a knee appears — that knee is the right operating point for max
throughput-feasible.

**Mitigations:**
- Lower concurrency (match `--max-num-seqs` to actual demand).
- Quantize KV cache (`--kv-cache-dtype fp8`) — halves KV memory at small
  accuracy cost, often pushes infeasible cells into feasible.
- Larger / faster GPU (more HBM bandwidth, not just more capacity).
- Multi-GPU (tensor parallel) to spread KV across cards.

## When to reduce context length

- TTFT p95 violates SLO even at concurrency=1.
- VRAM peak is fine but tail latency is unacceptable.
- Often the cheapest mitigation: a tokenizer-aware chunking step on the
  client side gets you most of the way back into feasibility.

## When to reduce concurrency

- VRAM peak is in the marginal band (22–24 GB) at moderate context.
- TPOT p95 is much higher than TPOT p50 (queueing-driven tail).
- Useful when peak load is bursty: cap `max-num-seqs` so the tail is
  predictable, even if average throughput drops.

## When to use FP8 KV cache

- VRAM peak crosses the marginal threshold *only* at high (ctx × conc).
- You're context-driven (long docs) more than quality-driven.
- Run a sweep with `KV_CACHE_DTYPE=fp8` — usually shifts the OOM boundary
  outward by ~2× along the concurrency axis at long context.

## When multi-GPU becomes necessary

- Largest feasible context length is below your application's needs even
  with FP8 KV.
- Aggregate throughput plateaus before you reach SLO-feasible concurrency.
- 32B+ class models (weights alone don't fit comfortably alongside KV).
- TP=2 across two 24 GB cards roughly doubles available KV budget and
  halves prefill latency, at the cost of inter-GPU NCCL overhead during
  decode.
