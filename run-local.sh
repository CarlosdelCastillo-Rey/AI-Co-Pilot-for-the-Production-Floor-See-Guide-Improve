#!/usr/bin/env bash
# Run vision-ops-backend (8000) + vision-ops-app (3000) together.
# Stop with Ctrl+C — releases the webcam and both servers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/vision-ops-backend"
FRONTEND_DIR="$ROOT/vision-ops-app"

BACKEND_PID=""
FRONTEND_PID=""
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

log() {
  printf '\033[1;36m[run-local]\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33m[run-local]\033[0m %s\n' "$*" >&2
}

free_port() {
  local port="$1"
  local label="$2"

  if ! lsof -ti ":${port}" >/dev/null 2>&1; then
    return 0
  fi

  log "Port ${port} (${label}) in use — stopping PID(s): $(lsof -ti ":${port}" | tr '\n' ' ')"
  lsof -ti ":${port}" | xargs kill -TERM 2>/dev/null || true
  sleep 0.5
  if lsof -ti ":${port}" >/dev/null 2>&1; then
    lsof -ti ":${port}" | xargs kill -9 2>/dev/null || true
  fi
}

free_all_ports() {
  pkill -f "vision_ops_backend.main:app" 2>/dev/null || true
  pkill -f "next dev" 2>/dev/null || true
  free_port "$BACKEND_PORT" "backend"
  free_port "$FRONTEND_PORT" "frontend"
}

cleanup() {
  log "Shutting down..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  free_all_ports
  wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    warn "Missing required command: $1"
    exit 1
  fi
}

require_cmd uv
require_cmd npm
require_cmd lsof

if [[ ! -d "$BACKEND_DIR" || ! -d "$FRONTEND_DIR" ]]; then
  warn "Run this script from the repository root (vision-ops-backend and vision-ops-app must exist)."
  exit 1
fi

# Frontend env
if [[ ! -f "$FRONTEND_DIR/.env.local" ]]; then
  if [[ -f "$FRONTEND_DIR/.env.local.example" ]]; then
    cp "$FRONTEND_DIR/.env.local.example" "$FRONTEND_DIR/.env.local"
    log "Created vision-ops-app/.env.local from .env.local.example"
  fi
fi

die() {
  warn "$1"
  exit 1
}

ensure_face_models() {
  local sface="$ROOT/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
  local yunet="$ROOT/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  local installer="$ROOT/models/install_face_models.sh"

  if [[ ! -f "$installer" ]]; then
    die "Missing $installer — pull latest repo (install script must be in git)."
  fi
  chmod +x "$installer" 2>/dev/null || true

  if [[ -f "$sface" && -f "$yunet" ]]; then
    log "Face models OK (local cache in models/)"
    return 0
  fi

  log "First run: downloading face models locally (~40 MB, not in git)..."
  log "See models/README.md for details."
  bash "$installer" || die "Model install failed. Run: ./models/install_face_models.sh"

  [[ -f "$sface" && -f "$yunet" ]] || die "Models still missing after install. Check network / Hugging Face access."
}

ensure_face_models

log "Installing backend dependencies (uv sync)..."
(cd "$BACKEND_DIR" && uv sync)

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  log "Installing frontend dependencies (npm install)..."
  (cd "$FRONTEND_DIR" && npm install)
fi

log "Clearing ports ${BACKEND_PORT} (backend) and ${FRONTEND_PORT} (frontend)..."
free_all_ports

log "Starting backend  → http://localhost:${BACKEND_PORT}"
(
  cd "$BACKEND_DIR"
  exec uv run uvicorn vision_ops_backend.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT"
) 2>&1 | sed 's/^/[backend] /' &
BACKEND_PID=$!

sleep 2

log "Starting frontend → http://localhost:${FRONTEND_PORT}"
(
  cd "$FRONTEND_DIR"
  exec npm run dev
) 2>&1 | sed 's/^/[frontend] /' &
FRONTEND_PID=$!

echo ""
log "Ready:"
log "  Live UI:  http://localhost:${FRONTEND_PORT}/live"
log "  Identity: http://localhost:${FRONTEND_PORT}/identity"
log "  API:      http://localhost:${BACKEND_PORT}/health"
log "  Press Ctrl+C to stop both (and release the webcam)."
echo ""

wait "$BACKEND_PID" "$FRONTEND_PID"
