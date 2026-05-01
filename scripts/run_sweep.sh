#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run a benchmark sweep against an already-running vLLM server, then analyze.
#
# Assumes:
#   - vLLM was started via scripts/start_vllm_qwen{14,32}b_awq.sh and is healthy
#     on http://localhost:8000  (override via SERVER_URL).
#   - The client-side venv has requirements.txt installed.
#
# Usage:
#   bash scripts/run_sweep.sh                              # 14B, default config
#   CONFIG=configs/qwen32b_awq.yaml bash scripts/run_sweep.sh
# -----------------------------------------------------------------------------
set -euo pipefail

CONFIG="${CONFIG:-configs/qwen14b_awq.yaml}"
OUT_CSV="${OUT_CSV:-results/benchmark_results.csv}"
SERVER_URL="${SERVER_URL:-http://localhost:8000/v1}"

echo "----------------------------------------------------------------------"
echo "Benchmark sweep:"
echo "  config = ${CONFIG}"
echo "  server = ${SERVER_URL}"
echo "  csv    = ${OUT_CSV}"
echo "----------------------------------------------------------------------"

# Quick liveness probe so we don't burn 5 minutes waiting for a dead server.
if ! curl -fsS --max-time 5 "${SERVER_URL%/v1}/v1/models" >/dev/null 2>&1; then
  echo "[run_sweep] ERROR: cannot reach ${SERVER_URL}/models  (is vLLM up?)" >&2
  exit 1
fi

python -m benchmark.sweep_benchmark \
  --config "${CONFIG}" \
  --out-csv "${OUT_CSV}"

python -m benchmark.analyze_results "${OUT_CSV}" --out-dir "$(dirname "${OUT_CSV}")"

echo "[run_sweep] done -> $(dirname "${OUT_CSV}")/"
