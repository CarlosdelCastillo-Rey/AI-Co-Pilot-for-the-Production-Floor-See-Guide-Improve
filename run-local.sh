#!/usr/bin/env bash
# Run unified vision-ops-backend (:8000) + vision-ops-app (:3000).
# HAR v2 session audit, Re-ID, HITL, alerts, and advisor run in-process on the backend.
# Also sets up Ollama for the Advisor LLM: brew install --cask ollama, open Ollama.app, ollama pull llama3.1.
# Stop with Ctrl+C — releases the webcam and all servers (and ollama serve if we started it).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/vision-ops-backend"
FRONTEND_DIR="$ROOT/vision-ops-app"
HAR_RESEARCH_DIR="$ROOT/har-research"

BACKEND_PID=""
FRONTEND_PID=""
OLLAMA_PID=""
OLLAMA_STARTED_BY_SCRIPT=false
OLLAMA_HOST="${VISIONOPS_OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${VISIONOPS_OLLAMA_MODEL:-llama3.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

export HAR_V2_SESSION_ENABLED="${HAR_V2_SESSION_ENABLED:-true}"
export HAR_V2_MIN_CONFIDENCE="${HAR_V2_MIN_CONFIDENCE:-0.25}"
export HAR_ACTIVITY_INGEST_ENABLED="${HAR_ACTIVITY_INGEST_ENABLED:-true}"
export VISIONOPS_HAR_REID_MATCH_THRESHOLD="${VISIONOPS_HAR_REID_MATCH_THRESHOLD:-0.95}"
export VISIONOPS_HAR_SESSION_ARTIFACTS_DIR="${VISIONOPS_HAR_SESSION_ARTIFACTS_DIR:-${BACKEND_DIR}/data/har_sessions}"
export HAR_LIVE_PER_PERSON_MODE="${HAR_LIVE_PER_PERSON_MODE:-true}"
export HAR_CHECKPOINT_DIR="${HAR_CHECKPOINT_DIR:-har-research/checkpoints}"

OLLAMA_AUTO_INSTALL="${OLLAMA_AUTO_INSTALL:-true}"
OLLAMA_AUTO_PULL="${OLLAMA_AUTO_PULL:-true}"

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
  if [[ "$OLLAMA_STARTED_BY_SCRIPT" == "true" && -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID" 2>/dev/null || true
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

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  if [[ -f "$BACKEND_DIR/.env.example" ]]; then
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
    log "Created vision-ops-backend/.env from .env.example"
  fi
fi

die() {
  warn "$1"
  exit 1
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local i
  for i in $(seq 1 60); do
    if curl -sf --connect-timeout 1 "$url" >/dev/null 2>&1; then
      log "${name} ready → ${url}"
      return 0
    fi
    sleep 0.5
  done
  warn "${name} did not respond at ${url} (continuing anyway)"
  return 1
}

ensure_har_v2_dirs() {
  mkdir -p "${VISIONOPS_HAR_SESSION_ARTIFACTS_DIR}"
  log "HAR v2 artifacts → ${VISIONOPS_HAR_SESSION_ARTIFACTS_DIR}"
  if [[ -d "$HAR_RESEARCH_DIR" ]]; then
    log "HAR research pipeline → ${HAR_RESEARCH_DIR}"
  fi
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

ollama_on_path() {
  if [[ -d "/Applications/Ollama.app/Contents/Resources" ]]; then
    export PATH="/Applications/Ollama.app/Contents/Resources:${PATH}"
  fi
  command -v ollama >/dev/null 2>&1
}

ollama_healthy() {
  curl -sf --connect-timeout 2 "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1
}

ollama_has_model() {
  ollama_on_path || return 1
  local base="${OLLAMA_MODEL%%:*}"
  ollama list 2>/dev/null | grep -qE "${base}(:|latest)?" || return 1
}

# Homebrew *formula* ollama often lacks llama-server; the *cask* ships the full app.
ollama_formula_without_server() {
  if ! command -v brew >/dev/null 2>&1; then
    return 1
  fi
  if brew list --cask ollama >/dev/null 2>&1; then
    return 1
  fi
  if ! brew list ollama >/dev/null 2>&1; then
    return 1
  fi
  local prefix
  prefix="$(brew --prefix 2>/dev/null)/Cellar/ollama"
  if [[ -d "$prefix" ]] && ! find "$prefix" -name llama-server -type f 2>/dev/null | grep -q .; then
    return 0
  fi
  if ollama_on_path && ollama list 2>&1 | grep -q "llama-server binary not found"; then
    return 0
  fi
  return 1
}

repair_ollama_brew_install() {
  if [[ "$OLLAMA_AUTO_INSTALL" != "true" ]]; then
    return 0
  fi
  if ! ollama_formula_without_server; then
    return 0
  fi
  warn "Detected broken Homebrew formula 'ollama' (missing llama-server)."
  warn "Replacing with official Ollama.app (brew install --cask ollama)..."
  brew uninstall ollama 2>/dev/null || true
  brew install --cask ollama || return 1
  ollama_on_path
}

install_ollama_cli() {
  repair_ollama_brew_install || true
  ollama_on_path && return 0

  if [[ "$OLLAMA_AUTO_INSTALL" != "true" ]]; then
    warn "Ollama not found. Install: brew install --cask ollama  OR  https://ollama.com/download"
    return 1
  fi
  if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew not found — install Ollama from https://ollama.com/download"
    return 1
  fi
  log "Installing Ollama desktop app (brew install --cask ollama)..."
  if ! brew install --cask ollama; then
    warn "brew install --cask ollama failed — try https://ollama.com/download"
    return 1
  fi
  ollama_on_path
}

start_ollama_server() {
  if ollama_healthy; then
    return 0
  fi

  # macOS: Ollama.app bundles llama-server — prefer over broken `ollama serve` from formula
  if [[ "$(uname -s)" == "Darwin" ]] && [[ -d "/Applications/Ollama.app" ]]; then
    log "Starting Ollama.app (menu bar) → ${OLLAMA_HOST} ..."
    open -a Ollama >/dev/null 2>&1 || true
    local i
    for i in $(seq 1 45); do
      if ollama_healthy; then
        return 0
      fi
      sleep 1
    done
  fi

  if ! ollama_on_path; then
    return 1
  fi
  if ollama_formula_without_server; then
    warn "Skip ollama serve — formula install has no llama-server. Use: brew uninstall ollama && brew install --cask ollama"
    return 1
  fi

  log "Starting Ollama server (ollama serve) → ${OLLAMA_HOST} ..."
  ollama serve >/dev/null 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_STARTED_BY_SCRIPT=true

  local i
  for i in $(seq 1 30); do
    if ollama_healthy; then
      return 0
    fi
    sleep 1
  done
  warn "Ollama did not become ready on ${OLLAMA_HOST}"
  return 1
}

ollama_inference_ok() {
  local out
  out="$(curl -sf --max-time 90 "${OLLAMA_HOST%/}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${OLLAMA_MODEL}\",\"prompt\":\"ok\",\"stream\":false}" 2>&1)" || return 1
  echo "$out" | grep -q '"response"' && ! echo "$out" | grep -q 'llama-server binary not found'
}

pull_ollama_model() {
  if ! ollama_on_path; then
    return 1
  fi
  if ollama_has_model; then
    log "Ollama model present: ${OLLAMA_MODEL}"
    return 0
  fi
  if [[ "$OLLAMA_AUTO_PULL" != "true" ]]; then
    warn "Model missing — run: ollama pull ${OLLAMA_MODEL}"
    return 1
  fi
  log "Pulling Ollama model ${OLLAMA_MODEL} (first run may take several minutes)..."
  if ollama pull "${OLLAMA_MODEL}"; then
    log "Ollama model ready: ${OLLAMA_MODEL}"
    return 0
  fi
  warn "ollama pull ${OLLAMA_MODEL} failed — check disk space and network"
  return 1
}

ensure_ollama() {
  log "Setting up Advisor LLM (Ollama + ${OLLAMA_MODEL})..."

  install_ollama_cli || return 1
  start_ollama_server || return 1
  pull_ollama_model || return 1

  if ollama_healthy && ollama_has_model && ollama_inference_ok; then
    log "Advisor LLM ACTIVE → ${OLLAMA_HOST} (${OLLAMA_MODEL})"
    return 0
  fi
  if ollama_formula_without_server; then
    warn "Fix: brew uninstall ollama && brew install --cask ollama && open -a Ollama"
  elif ollama_healthy && ollama_has_model; then
    warn "Ollama is up but inference test failed — open Ollama.app or reinstall cask"
  fi
  warn "Advisor LLM setup incomplete — /alerts may show DOWN"
  return 1
}

WEBCAM_ENABLED="${WEBCAM_ENABLED:-false}"
export WEBCAM_ENABLED
export FACE_ENABLED="${FACE_ENABLED:-false}"

if [[ "$WEBCAM_ENABLED" == "true" ]]; then
  ensure_face_models
else
  log "Webcam disabled (WEBCAM_ENABLED=false) — MacBook camera will not open"
fi

log "Installing backend dependencies (uv sync --extra har)..."
(cd "$BACKEND_DIR" && uv sync --extra har)

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  log "Installing frontend dependencies (npm install)..."
  (cd "$FRONTEND_DIR" && npm install)
fi

ensure_ollama || true

ensure_har_v2_dirs

log "Clearing ports ${BACKEND_PORT} (backend) and ${FRONTEND_PORT} (frontend)..."
free_all_ports

log "Starting backend  → http://localhost:${BACKEND_PORT}"
(
  cd "$BACKEND_DIR"
  exec uv run uvicorn vision_ops_backend.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT" --timeout-graceful-shutdown 3
) 2>&1 | sed 's/^/[backend] /' &
BACKEND_PID=$!

wait_for_health "http://127.0.0.1:${BACKEND_PORT}/health" "Backend" || true
wait_for_health "http://127.0.0.1:${BACKEND_PORT}/api/har/v2/sessions?limit=1" "HAR v2 API" || true

log "Starting frontend → http://localhost:${FRONTEND_PORT}"
(
  cd "$FRONTEND_DIR"
  exec npm run dev
) 2>&1 | sed 's/^/[frontend] /' &
FRONTEND_PID=$!

echo ""
log "Ready:"
log "  Login:           http://localhost:${FRONTEND_PORT}/login"
log "  Analytics:       http://localhost:${FRONTEND_PORT}/analytics"
log "  Live streams:    http://localhost:${FRONTEND_PORT}/live  (eval + model probes + session audit)"
log "  Person HITL:     http://localhost:${FRONTEND_PORT}/har-hitl  (sessions, registry, review, tuning)"
log "  Timeline:        http://localhost:${FRONTEND_PORT}/timeline"
log "  Alert rules:     http://localhost:${FRONTEND_PORT}/alerts"
log "  Settings:        http://localhost:${FRONTEND_PORT}/settings"
log "  API:             http://localhost:${BACKEND_PORT}/health"
log "  HAR v2 API:      http://localhost:${BACKEND_PORT}/api/har/v2/sessions"
log "  Advisor:         http://localhost:${FRONTEND_PORT}/alerts  (Ollama status chip)"
if [[ "$HAR_V2_SESSION_ENABLED" == "true" ]]; then
  log "  HAR v2 logging:  ON (bench + live → ${VISIONOPS_HAR_SESSION_ARTIFACTS_DIR})"
else
  log "  HAR v2 logging:  OFF (set HAR_V2_SESSION_ENABLED=true to enable)"
fi
if [[ "$HAR_LIVE_PER_PERSON_MODE" == "true" ]]; then
  log "  Live cameras:    per-person YOLO+ByteTrack ON (cam-har-01…02)"
else
  log "  Live cameras:    scene-level (set HAR_LIVE_PER_PERSON_MODE=true)"
fi
if ollama_healthy && ollama_has_model; then
  log "  Ollama:   ACTIVE ${OLLAMA_HOST} (${OLLAMA_MODEL})"
elif ollama_healthy; then
  log "  Ollama:   server up — model missing (run: ollama pull ${OLLAMA_MODEL})"
else
  log "  Ollama:   DOWN — set OLLAMA_AUTO_INSTALL=true or install from https://ollama.com"
fi
log "  Demo login: admin@visionops.local / admin123"
if [[ "$WEBCAM_ENABLED" == "true" ]]; then
  log "  Webcam:          enabled (set WEBCAM_ENABLED=false to skip MacBook camera)"
else
  log "  Webcam:          disabled (mock HAR videos only)"
fi
log ""
log "  HAR workflow: Live Streams (play video) → Person HITL (sessions, registry, review)"
log "  Press Ctrl+C to stop all servers."
echo ""

wait "$BACKEND_PID" "$FRONTEND_PID"
