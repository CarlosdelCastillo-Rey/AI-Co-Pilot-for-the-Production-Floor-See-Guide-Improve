# VisionOps HAR Research

Canonical **machine-learning pipeline** for per-person Human Activity Recognition on the InHARD 12-class industrial dataset. Trained checkpoints are consumed at runtime by `vision-ops-backend` (`./run-local.sh`).

## Quick start

```bash
# From repo root — research Python env
uv sync --all-groups
cd har-research
jupyter lab 00_Pipeline_Run_All.ipynb
```

## Pipeline (current notebooks)

| Notebook | Purpose |
|----------|---------|
| `00_Pipeline_Run_All.ipynb` | Master orchestrator — data → embeddings → train → eval |
| `01_Data_and_Strategy.ipynb` | Clip inventory, class balance, top-down crop strategy |
| `02_Embedding_Extraction.ipynb` | Frozen V-JEPA2 + DINOv2 with attention pooling |
| `03_Train_HAR_Head.ipynb` | MLP head — Focal Loss, Mixup, subject split |
| `04_Analysis_and_Visualization.ipynb` | UMAP, confusion matrix, frame strips |
| `05_Mock_Video_Eval.ipynb` | Mock industrial MP4 eval + session logging |
| `08_Session_Log_Review.ipynb` | Session audit, track labels, Re-ID review |

## Layout

```text
har-research/
├── lib/              # Shared modules (imported by notebooks + backend)
├── checkpoints/      # Production weights (.pt + .json metadata)
└── outputs/          # Embeddings, metrics, charts, har_sessions/ (gitignored)
```

## Production checkpoints

| Checkpoint stem | Backbone | Demo role |
|-----------------|----------|-----------|
| `har_vjepa_12c_topdown_allavail` | V-JEPA 2 ViT-L | Live model `v2-vjepa` |
| `har_dinov2_12c_topdown_allavail` | DINOv2 ViT-L | Live model `v2-dinov2` |

Live UI uses a **4-slot mock wall** (`cam-har-mock-0…3`) with ByteTrack per-person mode. Legacy per-model cameras `cam-har-01` / `cam-har-02` remain in the DB for batch probes.

## Link to demo

| Research output | Runtime consumer |
|-----------------|------------------|
| `checkpoints/*.pt` | `HAR_CHECKPOINT_DIR` in backend |
| `outputs/har_sessions/` | HITL artifacts mirror (`VISIONOPS_HAR_SESSION_ARTIFACTS_DIR`) |
| `lib/` crop + train utils | `vision_ops_backend.vision.har` inference |

OpenAPI HAR v2: `GET /api/har/v2/sessions` · UI: http://localhost:3000/har-hitl
