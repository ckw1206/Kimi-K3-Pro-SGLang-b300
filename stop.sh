#!/usr/bin/env bash
set -euo pipefail

# Stop the container started by ./start.sh. Idempotent.
# The stopped container is left in place so `docker logs` still works;
# the next ./start.sh removes it.

CONTAINER_NAME="kimi-k3-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} does not exist; nothing to stop"
elif docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping container ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null
  echo "[$(date -Is)] container ${CONTAINER_NAME} stopped" >> "${LOG_FILE}"
  echo "Stopped ${CONTAINER_NAME}."
else
  echo "Container ${CONTAINER_NAME} is not running"
fi
rm -f "${PID_FILE}"
