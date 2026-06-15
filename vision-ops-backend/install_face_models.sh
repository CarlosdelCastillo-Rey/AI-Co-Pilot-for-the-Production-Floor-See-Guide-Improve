#!/usr/bin/env bash
# Download OpenCV YuNet + SFace ONNX models into vision-ops-backend/weights/ (not committed to git).
# Called automatically by ./run-local.sh on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$ROOT/weights"
SFACE_DIR="$WEIGHTS_DIR/face_recognition_sface"
YUNET_DIR="$WEIGHTS_DIR/face_detection_yunet"
SFACE_ONNX="$SFACE_DIR/face_recognition_sface_2021dec.onnx"
YUNET_ONNX="$YUNET_DIR/face_detection_yunet_2023mar.onnx"

log() { printf '\033[1;36m[install-face-models]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install-face-models]\033[0m %s\n' "$*" >&2; }
die() { warn "$1"; exit 1; }

need_file() { [[ -f "$1" ]]; }

if need_file "$SFACE_ONNX" && need_file "$YUNET_ONNX"; then
  log "Models already present — skipping download."
  exit 0
fi

mkdir -p "$WEIGHTS_DIR"

if ! command -v curl >/dev/null 2>&1; then
  die "curl is required to download models (or install Hugging Face CLI: curl -LsSf https://hf.co/cli/install.sh | bash)"
fi

download_hf() {
  local repo_id="$1"
  local local_dir="$2"
  if command -v hf >/dev/null 2>&1; then
    log "hf download $repo_id → $local_dir"
    hf download "$repo_id" --local-dir "$local_dir"
    return 0
  fi
  log "Hugging Face CLI (hf) not found — using huggingface_hub via Python..."
  if command -v uv >/dev/null 2>&1; then
    uv run --with huggingface-hub python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo_id}", local_dir="${local_dir}")
PY
    return 0
  fi
  if python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo_id}", local_dir="${local_dir}")
PY
    return 0
  fi
  die "Install Hugging Face tools: curl -LsSf https://hf.co/cli/install.sh | bash  OR  pip install huggingface-hub"
}

log "First-time setup: downloading ~40 MB of ONNX weights from Hugging Face..."
log "These files stay on your machine only (gitignored in vision-ops-backend/weights/)."

download_hf "opencv/face_detection_yunet" "$YUNET_DIR"
download_hf "opencv/face_recognition_sface" "$SFACE_DIR"

need_file "$SFACE_ONNX" || die "Missing SFace ONNX after download: $SFACE_ONNX"
need_file "$YUNET_ONNX" || die "Missing YuNet ONNX after download: $YUNET_ONNX"

log "Done."
log "  SFace: $SFACE_ONNX"
log "  YuNet: $YUNET_ONNX"
