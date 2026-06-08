# notebooks_v2 — HAR Pipeline v2 (parallel to `notebooks/`)

Run this pipeline **in parallel** while `notebooks/` (v1) finishes a long embedding job.

## What changed in v2

| Fix | v1 | v2 |
|-----|----|----|
| Classes | Often 14 (incl. blocked) | **12 trainable** (`Assemble` / `No action` excluded) |
| Embeddings | Full InHARD clip | **YOLO person crops** (matches live inference) |
| Train split | Random 80/20 | **Subject-held-out** by default |
| Loss | Plain CE | **Class-weighted** CE |
| Inference | Always argmax | **Confidence gate** (`min_confidence=0.25`) |
| Human labels | — | **Notebook 07** carousel UI → merged in step 02 |
| Outputs | `notebooks/outputs/` | **`notebooks_v2/outputs/`** (isolated) |

## Quick start

```bash
cd notebooks_v2
# In Jupyter: open 00_Pipeline_Run_All.ipynb
```

Mount InHARD external drive or set:

```bash
export INHARD_ROOT="/path/to/01-InHARD"
```

## Notebook order

| Notebook | Purpose |
|----------|---------|
| `00_Pipeline_Run_All.ipynb` | Full rebuild (02→04) |
| `00_b_Pipeline_Resume.ipynb` | Skip 02/03 if config fingerprint matches |
| `01_Data_and_Strategy.ipynb` | 12-class sample preview |
| `01b_InHARD_Explore.ipynb` | EDA → `outputs/inhard_eda/` |
| `02_Embedding_Extraction.ipynb` | Crop-aligned + optional human merge |
| `03_Train_HAR_Head.ipynb` | Weighted MLP + subject split |
| `04_PerPerson_Eval.ipynb` | Mock video + session logs for HITL |
| `05_Live_Camera_Demo.ipynb` | OpenCV live UI |
| `06_Model_and_Session_Analysis.ipynb` | Metrics / confusion matrix |
| **`07_Human_Label_Review.ipynb`** | **Tinder-style review UI** |

## Human-in-the-loop loop

1. Run **04** (or live **05**) → crops saved under `outputs/har_sessions/`
2. Open **07** → review queue (low confidence first)
3. Re-run **02** with `include_human_labels=True` (default)
4. Re-run **03** → checkpoint updates
5. Re-eval **04** / **06**

Labels CSV: `outputs/human_labels/labels.csv`

## Default checkpoint

`checkpoints/har_vjepa_v2_12c_100each.pt`

## Review v1 sessions

Notebook 07 also scans `notebooks/outputs/har_sessions/` if present.
