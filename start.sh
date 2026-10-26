#!/usr/bin/env bash
set -euo pipefail

# Kimi K3 on SGLang (8x B300 per node, 1/2/3 nodes)
#
# Recipe from the SGLang cookbook (B300 cells):
#   https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3
#
# Layouts (every node runs ./start.sh with its own NODE_RANK):
#   1 node  (8x B300):  TP8, symm-mem
#   2 nodes (16x B300): TP16, EP16, attn-DP2
#   3 nodes (24x B300): TP24, EP24, attn-DP3
# Only rank 0 serves the API; nodes reach each other on DIST_PORT and the
# high-speed fabric. First boot downloads the weights into HF_CACHE.

# Load optional .env overrides. Shell env wins; .env fills gaps; defaults last.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

MODEL_ID="${MODEL_ID:-moonshotai/kimi-k3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
DIST_PORT="${DIST_PORT:-20000}"
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_DEBUG="${NCCL_DEBUG:-}"
SHM_SIZE="${SHM_SIZE:-32g}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Free-form extra SGLang flags, appended LAST (argparse last-wins):
#   EXTRA_ARGS="--max-running-requests 256" ./start.sh
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

case "${NNODES}" in
  1|2|3) : ;;
  *) echo "NNODES must be 1, 2 or 3, got '${NNODES}'"; exit 1 ;;
esac
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "NODE_RANK must be 0..$((NNODES - 1)) for NNODES=${NNODES}, got '${NODE_RANK}'"; exit 1
fi
# Strip leading zeros so 08 is not read as octal.
NODE_RANK=$((10#${NODE_RANK}))
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be 0..$((NNODES - 1)) for NNODES=${NNODES}, got '${NODE_RANK}'"; exit 1
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is required when NNODES > 1 (IP of the rank-0 node)"; exit 1
fi
GPU_COUNT="$(tr ',' '\n' <<< "${GPU_ID}" | grep -c '[0-9]' || true)"
if (( GPU_COUNT != 8 )); then
  echo "GPU_ID='${GPU_ID}' gives ${GPU_COUNT} GPUs; need exactly 8 (one B300 node)"; exit 1
fi

# Cookbook starting points: 0.82 on 1 node, 0.92 on 2/3 nodes.
if [[ -z "${MEM_FRACTION:-}" ]]; then
  if (( NNODES == 1 )); then MEM_FRACTION=0.82; else MEM_FRACTION=0.92; fi
fi

TP=$((8 * NNODES))

if (( NNODES > 1 )); then
  DIST_ARGS=(
    --ep-size "${TP}"
    --attn-dp-size "${NNODES}"
    --enable-dp-lm-head
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --dist-init-addr "${MASTER_ADDR}:${DIST_PORT}"
    --moe-a2a-backend megamoe
    --moe-runner-backend deep_gemm
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --mamba-radix-cache-strategy extra_buffer_lazy
  )
  # Cross-node MoE all-to-all goes over RDMA: expose the IB devices and allow pinning.
  DOCKER_DIST_ARGS=(-e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480
    --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK)
else
  DIST_ARGS=(--enable-symm-mem)
  DOCKER_DIST_ARGS=()
fi

DOCKER_NCCL_ENV=()
if [[ -n "${NCCL_SOCKET_IFNAME}" ]]; then
  DOCKER_NCCL_ENV+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" -e "GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
fi
if [[ -n "${NCCL_IB_HCA}" ]]; then
  DOCKER_NCCL_ENV+=(-e "NCCL_IB_HCA=${NCCL_IB_HCA}")
fi
if [[ -n "${NCCL_DEBUG}" ]]; then
  DOCKER_NCCL_ENV+=(-e "NCCL_DEBUG=${NCCL_DEBUG}")
fi

CONTEXT_ARGS=()
if [[ -n "${CONTEXT_LENGTH}" ]]; then
  CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
fi

CONTAINER_NAME="kimi-k3-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID}"
echo "Nodes: ${NNODES} (this node's rank: ${NODE_RANK})"
if (( NNODES > 1 )); then
  echo "GPUs: ${GPU_ID} (TP${TP} + EP${TP}, attn-DP${NNODES}, mem-fraction ${MEM_FRACTION})"
else
  echo "GPUs: ${GPU_ID} (TP${TP}, symm-mem, mem-fraction ${MEM_FRACTION})"
fi
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
if (( NODE_RANK == 0 )); then
  echo "Listening on ${HOST}:${PORT}"
  echo "Writing progress to ${LOG_FILE}"
fi

echo "[$(date -Is)] launching SGLang container" > "${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --cap-add SYS_NICE \
  --gpus "\"device=${GPU_ID}\"" \
  --shm-size "${SHM_SIZE}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${DOCKER_NCCL_ENV[@]+"${DOCKER_NCCL_ENV[@]}"} \
  ${DOCKER_DIST_ARGS[@]+"${DOCKER_DIST_ARGS[@]}"} \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size "${TP}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  ${DIST_ARGS[@]+"${DIST_ARGS[@]}"} \
  ${CONTEXT_ARGS[@]+"${CONTEXT_ARGS[@]}"} \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"} \
  >/dev/null

docker inspect -f '{{.Id}}' "${CONTAINER_NAME}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} ($(cat "${PID_FILE}"))"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Workers only join the rendezvous; the API serves on rank 0.
if (( NODE_RANK != 0 )); then
  echo "Worker container started; it will join rank 0 at ${MASTER_ADDR}:${DIST_PORT} (it must reach rank 0 within 10 minutes, so start rank 0 first)"
  echo "Readiness is reported by rank 0"
  echo "Log: docker logs -f ${CONTAINER_NAME}"
  exit 0
fi

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

# First boot downloads the weights into HF_CACHE; that alone can take a while.
echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Stop with ./stop.sh; logs: docker logs -f ${CONTAINER_NAME}"
