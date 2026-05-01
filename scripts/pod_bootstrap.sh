#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# pod_bootstrap.sh
#
# One-shot bootstrap for a freshly-rented Linux GPU pod (Runpod / Lambda /
# Vast). Does steps 2-4 from README's "Running on a rented GPU" section:
#
#   2) clone (or pull) the repo into /workspace
#   3) create venv + install client deps + vLLM
#   4) smoke-test: launch a tiny vLLM server, hit /v1/models, run
#      request_runner and gpu_monitor, then shut the server down cleanly.
#
# Idempotent: re-running skips work that's already done.
#
# Usage on a fresh pod (one of these):
#
#   # A) repo already cloned somewhere; just bootstrap:
#   bash scripts/pod_bootstrap.sh
#
#   # B) nothing on disk yet; one-liner that curls this script and runs it:
#   curl -fsSL https://raw.githubusercontent.com/xiaosi0517/llm-inference-feasibility/main/scripts/pod_bootstrap.sh | bash
#
# Knobs (env vars, all optional):
#   REPO_URL     git URL to clone (default: this repo)
#   BRANCH       branch to check out (default: main)
#   WORKDIR      where to clone (default: /workspace/llm-inference-feasibility)
#   VENV_DIR     venv path inside the repo (default: .venv)
#   SKIP_SMOKE   set to 1 to skip step 4 (e.g. to save a model download)
#   SMOKE_MODEL  smoke-test HF repo (default: Qwen/Qwen2.5-0.5B-Instruct)
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/xiaosi0517/llm-inference-feasibility.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace/llm-inference-feasibility}"
VENV_DIR="${VENV_DIR:-.venv}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SMOKE_PORT="${SMOKE_PORT:-8000}"
SMOKE_SERVED_NAME="qwen2.5-14b-awq"   # matches RunnerConfig default; do not change

# ---- pretty section banners --------------------------------------------------
_section() {
  echo ""
  echo "======================================================================"
  echo "  $*"
  echo "======================================================================"
}

_die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

# ---- step 2a: system sanity --------------------------------------------------
_section "Step 2a: system sanity (nvidia-smi, python, git)"

command -v git >/dev/null 2>&1 || _die "git not installed"
command -v python3 >/dev/null 2>&1 || _die "python3 not installed"
command -v nvidia-smi >/dev/null 2>&1 || _die "nvidia-smi not on PATH (no GPU drivers?)"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 --version

# ---- step 2b: clone or update the repo --------------------------------------
_section "Step 2b: clone or update repo at ${WORKDIR}"

if [[ -d "${WORKDIR}/.git" ]]; then
  echo "[bootstrap] repo already present, fetching latest on ${BRANCH}"
  git -C "${WORKDIR}" fetch --quiet origin "${BRANCH}"
  git -C "${WORKDIR}" checkout --quiet "${BRANCH}"
  git -C "${WORKDIR}" pull --ff-only --quiet origin "${BRANCH}"
else
  mkdir -p "$(dirname "${WORKDIR}")"
  git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${WORKDIR}"
fi

cd "${WORKDIR}"
git --no-pager log -1 --oneline

# ---- step 3: venv + python deps ---------------------------------------------
_section "Step 3: create venv and install requirements + vLLM (~5 min first time)"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade --quiet pip wheel

# client deps (fast)
pip install --quiet -r requirements.txt

# server dep (slow first time, ~2-3 GB of wheels). Skip if already there.
if ! python -c "import vllm" 2>/dev/null; then
  pip install --quiet "vllm>=0.6.3"
else
  echo "[bootstrap] vllm already installed: $(python -c 'import vllm; print(vllm.__version__)')"
fi

# ---- step 4: smoke test ------------------------------------------------------
if [[ "${SKIP_SMOKE}" == "1" ]]; then
  _section "Step 4: SKIPPED (SKIP_SMOKE=1)"
  echo "[bootstrap] done. Next: bash scripts/start_vllm_qwen14b_awq.sh"
  exit 0
fi

_section "Step 4: smoke test with ${SMOKE_MODEL} on port ${SMOKE_PORT}"

LOG_DIR="${WORKDIR}/results/server_logs"
mkdir -p "${LOG_DIR}"
SMOKE_LOG="${LOG_DIR}/smoke_$(date +%Y%m%d_%H%M%S).log"

# Launch a tiny *unquantized* server in the background. Reuse the existing
# launch script so we exercise the same code path as the real run; override
# the heavy knobs via env vars.
echo "[bootstrap] launching smoke server, log -> ${SMOKE_LOG}"
(
  MODEL="${SMOKE_MODEL}" \
  SERVED_NAME="${SMOKE_SERVED_NAME}" \
  PORT="${SMOKE_PORT}" \
  MAX_MODEL_LEN=2048 \
  MAX_NUM_SEQS=4 \
  QUANTIZATION=none \
  KV_CACHE_DTYPE=auto \
  GPU_MEM_UTIL=0.50 \
  bash scripts/start_vllm_qwen14b_awq.sh
) >"${SMOKE_LOG}" 2>&1 &
SERVER_PID=$!
echo "[bootstrap] smoke server pid=${SERVER_PID}"

# Always tear down, even on failure, so the pod doesn't leak a vLLM process.
_cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[bootstrap] stopping smoke server (pid=${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    # vLLM forks workers; give them a moment, then SIGKILL stragglers.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  fi
}
trap _cleanup EXIT

# Wait for /v1/models. First boot includes a model download, so be patient.
echo "[bootstrap] waiting for http://localhost:${SMOKE_PORT}/v1/models (up to 10 min)..."
READY=0
for i in $(seq 1 120); do
  if curl -fsS --max-time 3 "http://localhost:${SMOKE_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "[bootstrap] server ready after ~$((i * 5))s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[bootstrap] smoke server died early. Last 40 log lines:" >&2
    tail -n 40 "${SMOKE_LOG}" >&2 || true
    _die "smoke server failed to start (see ${SMOKE_LOG})"
  fi
  sleep 5
done
[[ "${READY}" == "1" ]] || { tail -n 40 "${SMOKE_LOG}" >&2; _die "timed out waiting for /v1/models"; }

echo ""
echo "[bootstrap] /v1/models response:"
curl -s "http://localhost:${SMOKE_PORT}/v1/models" | python -m json.tool

echo ""
echo "[bootstrap] running benchmark.request_runner (2 streamed requests)..."
python -m benchmark.request_runner

echo ""
echo "[bootstrap] running benchmark.gpu_monitor (2s NVML sample)..."
python -m benchmark.gpu_monitor

_section "Bootstrap complete"
cat <<EOF
Next steps (real sweep):

  # terminal 1: launch the production server
  . ${VENV_DIR}/bin/activate
  bash scripts/start_vllm_qwen14b_awq.sh

  # terminal 2: run the sweep + analyze
  . ${VENV_DIR}/bin/activate
  bash scripts/run_sweep.sh

Outputs land in ${WORKDIR}/results/.
EOF
