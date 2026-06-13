# VisionOps — Phase 0 scripts (legacy)

> **Not part of the current demo.** The runnable stack is `./run-local.sh` → `vision-ops-backend` + `vision-ops-app`. Canonical ML work lives in [`har-research/`](../har-research/).

Atomic Jupyter notebooks from the original VisionOps phase-0 plan. Each step writes to `outputs/` (gitignored). Notebooks `05_semantic_event` and `07_telegram_webhook` are obsolete — alerts use Strands + MailerSend/Telegram in the unified backend.

## Suggested order (historical)

```text
01_capture_stream.ipynb → 02_segment_frames.ipynb → 03_detect_and_track.ipynb
  → 04_build_event_buffer.ipynb → 06_generate_heatmap.ipynb
  → 08_vjepa_action_probe.ipynb (optional; needs GPU/HF weights)
```

## Prerequisites

```bash
uv sync --all-groups
cp .env.example .env   # optional REDIS_URL for notebook 04
uv run jupyter lab scripts/
```

See notebook table and smoke-test notes in the previous revision of this file; outputs land under repo-root `outputs/`.

For production HAR training and eval, use **`har-research/`** instead.
