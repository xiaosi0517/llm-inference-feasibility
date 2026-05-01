#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Start vLLM OpenAI-compatible server for Qwen2.5-32B-Instruct-AWQ on 1x 24GB GPU.
#
# 32B AWQ is at the very edge of a 24 GB card:
#   - weights resident:        ~17 GB
#   - leaves ~5-6 GB for KV cache + activations
# Defaults below are aggressive (FP8 KV, smaller max-model-len, max-num-seqs=4)
# to make this configuration even *boot*. If startup OOMs, drop GPU_MEM_UTIL or
# MAX_MODEL_LEN further.
# -----------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
SERVED_NAME="${SERVED_NAME:-qwen2.5-32b-awq}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Tighter limits than 14B because the 32B weights eat most of the 24 GB card.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

QUANTIZATION="${QUANTIZATION:-awq_marlin}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# FP8 KV is effectively required at 32B: it ~halves KV memory at minor accuracy
# cost. Override with KV_CACHE_DTYPE=auto to test the (likely OOM) fp16 path.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-half}"

ENFORCE_EAGER_FLAG=""
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ENFORCE_EAGER_FLAG="--enforce-eager"
fi

LOG_DIR="${LOG_DIR:-./results/server_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"

echo "----------------------------------------------------------------------"
echo "Launching vLLM (32B AWQ -- tight fit on 24 GB):"
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
