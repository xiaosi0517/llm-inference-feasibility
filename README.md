# Constraint-Driven LLM Inference Feasibility Benchmark

A small framework that answers one engineering question:

> Given a single 24 GB GPU, what model size, context length, and concurrency
> can we support under memory and latency constraints?

Deploys a quantized open-source LLM with vLLM, sweeps `(context_length × concurrency)`,
records TTFT / TPOT / latency / throughput / peak VRAM, and labels each cell as
**Feasible / Marginal / Infeasible**.

---

## Why context length and concurrency matter

The dominant runtime cost on a single GPU is the **KV cache**:

```text
KV ≈ 2 · num_layers · num_kv_heads · head_dim · context_len · concurrency · bytes_per_elem
```

Both context length and concurrency are linear multipliers, so a deployment
that comfortably handles `(4k ctx, conc=2)` may OOM at `(16k ctx, conc=8)` even
with the same model weights. The sweep makes that boundary visible.

Total VRAM ≈ `model weights + KV cache + activations + runtime overhead`.
Quantization (AWQ INT4 weights, FP8 KV cache) shifts the boundary outward.

See [`docs/bottleneck_analysis.md`](docs/bottleneck_analysis.md) for the
prefill-vs-decode breakdown and [`docs/decision_framework.md`](docs/decision_framework.md)
for what to change when a cell is OOM / marginal / TTFT-bound / TPOT-bound.

---

## Project layout

```text
llm-inference-feasibility/
├── configs/
│   ├── qwen14b_awq.yaml            # 14B sweep grid + feasibility thresholds
│   ├── qwen32b_awq.yaml            # 32B (tighter; FP8 KV recommended)
│   └── mock.yaml                   # tiny sweep against the mock server (CPU-only dev)
├── scripts/
│   ├── start_vllm_qwen14b_awq.sh   # launches vLLM OpenAI-compatible server
│   ├── start_vllm_qwen32b_awq.sh
│   └── run_sweep.sh                # liveness check + sweep + analyze
├── benchmark/
│   ├── request_runner.py           # async OpenAI client, TTFT/TPOT/latency
│   ├── gpu_monitor.py              # NVML background sampler
│   ├── sweep_benchmark.py          # driver: walks grid, applies 3-tier rule
│   └── analyze_results.py          # heatmaps, decision matrix, summary
├── tools/
│   └── mock_vllm_server.py         # OpenAI-compatible fake server, no GPU required
├── tests/                          # pytest unit tests (no GPU required)
├── docs/
│   ├── bottleneck_analysis.md
│   └── decision_framework.md
├── results/                        # CSV / JSON / PNG / MD outputs land here
├── requirements.txt                # client + dev deps (vLLM is server-side, separate)
├── pytest.ini
└── README.md
```

---

## Quick start (local dev with a real GPU)

```bash
pip install -r requirements.txt          # client deps
pip install "vllm>=0.6.3"                # server dep -- needs CUDA + recent torch

# terminal 1: start the server
bash scripts/start_vllm_qwen14b_awq.sh

# terminal 2: run the sweep + analyze
bash scripts/run_sweep.sh
```

Outputs (exactly four):

```text
results/
├── benchmark_results.csv     # one row per (ctx, conc) cell
├── feasibility_heatmap.png   # green / yellow / red grid
├── decision_matrix.md        # per-cell metrics + label table
└── summary_report.md         # best-feasible config, OOM boundary
```

---

## Local development without a GPU

vLLM needs CUDA, but the **client** half of this project is plain Python over HTTP
and doesn't care what's serving the OpenAI-compatible endpoint. So you can
exercise nearly all of the harness on a CPU-only laptop (Windows / macOS / Linux)
before paying for any cloud time. The recommended ladder:

### Stage 1 — unit tests (no network, no GPU, ~5 s)

The pure-logic pieces — feasibility classifier, percentile math, OOM regex,
config loader, plotting, report writers, monitor invariants — all have tests:

```bash
pip install -r requirements.txt
python -m pytest
```

43 tests across `tests/`. If these pass, your refactors haven't quietly broken
the heatmap labels or the CSV schema.

### Stage 2 — end-to-end against a mock vLLM server (~30 s)

`tools/mock_vllm_server.py` is a ~200-line aiohttp app that speaks the
OpenAI-compatible streaming protocol. It returns fake content with configurable
TTFT / TPOT delays and can be told to fail on demand (OOM, HTTP 500, timeout,
empty stream). Use it to validate the **whole pipeline** — request streaming,
percentile aggregation, CSV writing, heatmap rendering — without a GPU.

```bash
# Terminal 1: start the mock (defaults bind to 0.0.0.0:8000, served_name=qwen2.5-14b-awq)
python -m tools.mock_vllm_server

# Terminal 2: run a tiny 4-cell sweep against it
CONFIG=configs/mock.yaml bash scripts/run_sweep.sh
#   -- on Windows PowerShell where bash isn't available, run these directly:
#        python -m benchmark.sweep_benchmark --config configs/mock.yaml `
#          --out-csv results/benchmark_results.csv
#        python -m benchmark.analyze_results results/benchmark_results.csv `
#          --out-dir results
```

After the sweep you should see exactly the four MVP deliverables in `results/`:
`benchmark_results.csv`, `feasibility_heatmap.png`, `decision_matrix.md`,
`summary_report.md`. Every cell will be labeled **marginal** because the mock
can't report VRAM via NVML — that's the correct fallback path
(see `classify_feasibility` in `benchmark/sweep_benchmark.py`). To exercise
the failure-classification code paths, restart the mock with an env var:

```bash
MOCK_FAIL_MODE=oom         python -m tools.mock_vllm_server   # cells -> infeasible/oom
MOCK_FAIL_MODE=timeout     python -m tools.mock_vllm_server   # cells -> infeasible/timeout
MOCK_FAIL_MODE=http_500    python -m tools.mock_vllm_server   # cells -> infeasible/http_error
MOCK_FAIL_MODE=empty       python -m tools.mock_vllm_server   # cells -> infeasible/other
MOCK_FAIL_AFTER_REQ=4 MOCK_FAIL_MODE=oom python -m tools.mock_vllm_server  # fails partway
```

This is the only way to deterministically trigger every label in the
3-tier feasibility classifier. Real GPUs won't OOM on cue.

### Stage 3 — cheap cloud GPU smoke test (~$0.50, ~30 min)

Before kicking off the real 14B / 32B sweep, validate the GPU half on the
cheapest card you can rent (T4 / A10 on RunPod, Lambda, Vast.ai). Override the
model and grid to trivial sizes:

```bash
MODEL="Qwen/Qwen2.5-0.5B-Instruct" MAX_MODEL_LEN=2048 \
  bash scripts/start_vllm_qwen14b_awq.sh

# In configs/qwen14b_awq.yaml, temporarily shrink to:
#   sweep:
#     context_lengths: [1024]
#     concurrency_levels: [1]
bash scripts/run_sweep.sh
```

If this end-to-end run succeeds (vLLM starts, NVML reads memory, a non-empty
heatmap is produced), the only thing the production run depends on is having
enough VRAM — which is exactly what the sweep is designed to measure.

### Stage 4 — production sweep on the 24 GB card

See the next section.

---

## Running on a rented GPU (minimal-change runbook)

Everything below is the *only* configuration you should need to touch.
All paths are relative; no machine-specific identifiers in the code.

### 1. Provision

A 24 GB card (RTX 4090, A10G 24G, L4 24G, A5000) running:

- NVIDIA driver ≥ 535
- CUDA ≥ 12.1
- Python 3.10+
- Linux (vLLM does not run on native Windows; Linux box or WSL2 is required)

### 2. Install (two environments are cleanest)

```bash
git clone <your fork>  llm-inference-feasibility
cd llm-inference-feasibility

# Server env (heavy, has vllm + torch + cuda libs)
python -m venv .venv-server
. .venv-server/bin/activate
pip install "vllm>=0.6.3"
deactivate

# Client env (lightweight)
python -m venv .venv-client
. .venv-client/bin/activate
pip install -r requirements.txt
```

### 3. Knobs you may need to change

| Where                                | Default                          | When to change                                      |
|--------------------------------------|----------------------------------|-----------------------------------------------------|
| `scripts/start_vllm_*.sh` env vars   | `MODEL`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEM_UTIL`, `KV_CACHE_DTYPE` | different model, smaller card, OOM at startup |
| `configs/*.yaml: server.base_url`    | `http://localhost:8000/v1`       | server runs on a different host/port               |
| `configs/*.yaml: feasibility.*`      | 22 GB / 24 GB / 30 s p95         | different total VRAM (e.g. 16 GB or 48 GB card)    |
| `configs/*.yaml: sweep.*`            | `[1k,4k,8k,16k] × [1,2,4,8]`     | want a finer/coarser grid                          |

Typical override pattern:

```bash
# Pick a different HF repo at launch:
MODEL="some-org/my-model" \
MAX_MODEL_LEN=8192 \
GPU_MEM_UTIL=0.88 \
bash scripts/start_vllm_qwen14b_awq.sh

# Run the sweep against a remote server:
SERVER_URL=http://10.0.0.5:8000/v1 \
CONFIG=configs/qwen14b_awq.yaml \
bash scripts/run_sweep.sh
```

### 4. Sanity checks before kicking off a long sweep

```bash
# Server is alive and serving the expected model name?
curl -s http://localhost:8000/v1/models | python -m json.tool

# One end-to-end request works (this prints two RequestResult lines)?
python -m benchmark.request_runner

# NVML can read the GPU?
python -m benchmark.gpu_monitor
```

If all three pass, `bash scripts/run_sweep.sh` will work end-to-end.

---

## Feasibility classification

| Label         | Conditions                                                         |
|---------------|--------------------------------------------------------------------|
| Feasible      | no errors, peak VRAM < `feasible_max_mb`, p95 latency within SLO   |
| Marginal      | no errors, peak VRAM in `[feasible_max_mb, total_vram_mb)`, **or** p95 over SLO, **or** GPU monitor unavailable |
| Infeasible    | any OOM / timeout / HTTP / unknown error, **or** peak VRAM ≥ total |

Thresholds live in `configs/*.yaml` under `feasibility:`.

---

## Output schema (CSV)

| column                          | meaning                                            |
|---------------------------------|----------------------------------------------------|
| `context_length`                | requested input tokens (chat-template overhead excluded) |
| `concurrency`                   | in-flight requests during the cell                 |
| `n_requests` / `n_success`      | measured requests in the cell / how many succeeded |
| `success_rate`                  | n_success / n_requests                             |
| `ttft_ms_p50` / `ttft_ms_p95`   | time-to-first-token                                |
| `tpot_ms_p50` / `tpot_ms_p95`   | time-per-output-token (excludes first token)       |
| `latency_ms_p50` / `latency_ms_p95` | end-to-end request latency                     |
| `throughput_tps`                | aggregate output tokens / wall time of measured bursts |
| `gpu_mem_peak_mb` / `gpu_mem_mean_mb` | VRAM (NVML samples, summed across `device_indices`) |
| `feasibility`                   | feasible / marginal / infeasible                   |
| `failure_reason`                | ok / timeout / oom / http_error / other            |
| `burst_wall_time_s`             | total wall time of measured bursts (for throughput math) |

---

## Out of scope

Deliberately not included, to keep the project focused on the one question
above:

- Multi-engine comparisons (vLLM vs. SGLang vs. TGI).
- Multi-node / distributed sweeps.
- Web UI, dashboard, or live monitoring.
- Cost analysis or autoscaling.
- A plugin system for new metrics.

Knobs that *are* discussed in `docs/decision_framework.md` as ways to move the
feasibility boundary (FP8 KV cache, prefix caching, smaller model, multi-GPU)
can be exercised by re-running the existing sweep with different
`scripts/start_vllm_*.sh` env vars — no code changes required.
