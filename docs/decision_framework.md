# Decision framework

A flow you can follow once the heatmap is in front of you. Each branch points
to the smallest change that typically resolves the symptom.

## Symptom: OOM (any cell labeled `infeasible` with `failure_reason=oom`)

Tackle in this order — cheapest first:

1. **Lower `gpu_memory_utilization`** (e.g. `0.90 → 0.85`).
   Frees headroom for activation spikes. Costs a few hundred MB of KV pool.

2. **Reduce `max-num-seqs` to the largest concurrency you saw work.**
   Caps KV growth at the most-pressured axis without touching context.

3. **Reduce `max-model-len`** to the largest context length that worked.
   The KV pool is sized to `max-model-len * max-num-seqs`; both knobs matter.

4. **Switch to FP8 KV cache** (`--kv-cache-dtype fp8`).
   Roughly halves KV memory; minimal accuracy impact for most workloads.

5. **More aggressive weight quantization** (AWQ → AWQ Marlin → GPTQ INT4
   with smaller group size).
   Diminishing returns past AWQ INT4; usually skip and go to step 6.

6. **Smaller model** (32B → 14B → 7B).
   Biggest impact, last resort if quality budget allows.

7. **Multi-GPU (tensor parallel).**
   Real fix when the workload can't shrink. Two 24 GB cards ≈ one 48 GB card
   for KV, minus ~10–15% NCCL overhead during decode.

## Symptom: TTFT too high (prefill-bound)

1. **Reduce input length.** RAG: tighter top-k. Summarization: shrink earlier
   turns. Often the only thing that actually helps.

2. **Enable prefix caching** (`--enable-prefix-caching`).
   Huge win if your prompts share a long fixed prefix (system prompt, tool
   descriptions). Free for the cases where it applies; useless otherwise.

3. **Smaller model.** Prefill scales with `num_layers · d_model^2`; both
   shrink between size tiers.

4. **Faster GPU.** Prefill is FLOPs-bound; H100 ≈ 3× A100 ≈ 2× L40 in raw
   FP16/BF16 throughput.

## Symptom: TPOT too high (decode-bound)

1. **Reduce concurrency.** TPOT degrades linearly with concurrency once
   HBM bandwidth saturates; lowering `max-num-seqs` is the simplest knob.

2. **Reduce output length.** `max_tokens=128` instead of `512` in the client.
   Doesn't speed up per-token decode but caps tail latency.

3. **Faster GPU.** Decode is memory-bandwidth-bound; H100 (3.0 TB/s) vs
   A100 (2.0 TB/s) is roughly the speedup ratio.

4. **FP8 KV cache** also helps decode: smaller KV → less to read per token →
   higher TPOT.

## Symptom: VRAM peak in the marginal band (22–24 GB)

The cell works *now* but you have no margin for traffic spikes.

1. **Lower `gpu_memory_utilization`** to `0.85` and re-run the sweep.
   You'll lose throughput but gain stability.

2. **FP8 KV cache** if you haven't already.

3. **Cap concurrency at the highest cell that landed in the green band.**
   Marginal cells should not be production set-points unless you have an
   autoscaler that can shed load before VRAM crosses 24 GB.

## How to read the heatmap when you're done

- The largest **feasible** rectangle that contains your operational
  `(ctx, conc)` is your safe envelope.
- The **edge** of feasibility is where to invest engineering: prefix caching,
  KV quantization, or another GPU.
- **Marginal** cells are deployable only with an autoscaler that can drain
  load before VRAM saturates. Otherwise treat them as infeasible.
- **Infeasible** cells with `oom` reason: see this document, top.
- **Infeasible** cells with `timeout` reason: usually means prefill on a
  long context exceeded the request timeout under contention. Either raise
  `request_timeout_s` or accept that this corner of the grid isn't yours.
