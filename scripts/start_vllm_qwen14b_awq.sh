#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Start vLLM OpenAI-compatible server for Qwen2.5-14B-Instruct-AWQ on 1x 24GB GPU.
#
# Run in a *dedicated* conda/venv that has vLLM installed:
#     pip install "vllm>=0.6.3"
#
# All knobs are env-var overridable, e.g.:
#     MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4 bash scripts/start_vllm_qwen14b_awq.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- model / server identity -------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
SERVED_NAME="${SERVED_NAME:-qwen2.5-14b-awq}"   # name clients pass in `model=`
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# ---- capacity knobs (the sweep depends on these) -----------------------------
# MAX_MODEL_LEN must be >= the largest context_length we sweep.
# MAX_NUM_SEQS must be >= the largest concurrency we sweep.
# Spec sweep ceiling: ctx=16384, conc=8.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# ---- memory / quantization ---------------------------------------------------
# AWQ kernels: vLLM auto-selects "awq_marlin" on Ampere+ which is ~2x faster
# than the legacy "awq" kernel. Keep "awq_marlin" unless you hit kernel issues.
QUANTIZATION="${QUANTIZATION:-awq_marlin}"

# Fraction of total VRAM vLLM is allowed to use. 0.90 leaves ~2.4 GB headroom
# on a 24 GB card for activations + framework overhead. Lower this if you see
# OOM during prefill at long contexts.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# KV cache dtype: "auto" = fp16 on AWQ models. Set to "fp8" to ~halve KV memory
# at a small accuracy cost — useful when the sweep marks long-ctx cells OOM.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# ---- parallelism -------------------------------------------------------------
TP_SIZE="${TP_SIZE:-1}"           # tensor parallel; single GPU = 1
DTYPE="${DTYPE:-half}"            # AWQ activations run in fp16

# ---- misc --------------------------------------------------------------------
# Turn this on (export ENFORCE_EAGER=1) to disable CUDA graphs for cleaner OOM
# traces during debugging. Leave off in production runs — graphs help latency.
ENFORCE_EAGER_FLAG=""
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ENFORCE_EAGER_FLAG="--enforce-eager"
fi

# Log directory for the server's stdout/stderr.
LOG_DIR="${LOG_DIR:-./results/server_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"

echo "----------------------------------------------------------------------"
echo "Launching vLLM:"
echo "  model            = ${MODEL}"
echo "  served name      = ${SERVED_NAME}"
echo "  endpoint         = http://${HOST}:${PORT}"
echo "  max-model-len    = ${MAX_MODEL_LEN}"
echo "  max-num-seqs     = ${MAX_NUM_SEQS}"
echo "  quantization     = ${QUANTIZATION}"
echo "  kv-cache-dtype   = ${KV_CACHE_DTYPE}"
echo "  gpu-memory-util  = ${GPU_MEM_UTIL}"
echo "  tensor-parallel  = ${TP_SIZE}"
echo "  log              = ${LOG_FILE}"
echo "----------------------------------------------------------------------"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization "${QUANTIZATION}" \
  --dtype "${DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --no-enable-log-requests \
  ${ENFORCE_EAGER_FLAG} \
  2>&1 | tee "${LOG_FILE}"
