# Notebooks — unified HAR pipeline

Single research pipeline for InHARD → V-JEPA2 → per-person HAR → HITL → paper artifacts.

## Quick start

```bash
cd notebooks
jupyter notebook 00_Pipeline_Run_All.ipynb
```

Run cells **sequentially**. Set `CLIPS_PER_CLASS` in cell 2 (default **100** paper run · **5** quick · **1** smoke test).

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 00 | `00_Pipeline_Run_All.ipynb` | **Unified pipeline** — V-JEPA + DINOv2, analysis, comparison |
| 00b | `00_b_Pipeline_Resume.ipynb` | Resume with cache fingerprints |
| 01 | `01_Data_and_Strategy.ipynb` | Data strategy |
| 01b | `01b_InHARD_Explore.ipynb` | EDA InHARD |
| 02 | `02_Embedding_Extraction.ipynb` | V-JEPA2 embeddings only |
| 03 | `03_Train_HAR_Head.ipynb` | MLP head only |
| 04 | `04_PerPerson_Eval.ipynb` | Annotated eval video |
| 05 | `05_Live_Camera_Demo.ipynb` | OpenCV live demo |
| 06 | `06_Model_and_Session_Analysis.ipynb` | Holdout metrics + session charts |
| 07 | `07_Human_Label_Review.ipynb` | HITL review UI (Tinder-style) |
| 08 | `08_DINOv2_vs_VJEPA.ipynb` | *(deprecated — use 00)* |

**Avance 5:** `../Avance 5. Modelo final/Avance5.#56.ipynb` — ensembles on embeddings.

### Dual backbone (default in notebook 00)

`backbones=("vjepa", "dinov2")` trains both in one run (shared YOLO crops). DINOv2: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) `dinov2_vitl14`. Comparison → `outputs/backbone_comparison_<tag>.json`.

## Defaults (unified pipeline)

- **12 trainable classes** (excludes Assemble system, No action)
- **YOLO crop-aligned** embeddings (`embedding_mode=yolo_crop`) — matches live inference
- **Subject-held-out** split + **balanced class weights**
- **Confidence gate** 0.25 on live/eval
- **Human labels** merged from `outputs/human_labels/labels.csv`

## Outputs

| Path | Contents |
|------|----------|
| `outputs/` | Active pipeline artifacts |
| `outputs/archive/v1_fullclip/` | Baseline full-clip runs (paper comparison) |
| `outputs/archive/v2_crop_12c/` | Crop-aligned pilot runs (formerly `notebooks_v2`) |
| `outputs/paper/manifest.json` | Run index for `Paper/main.tex` |
| `outputs/har_sessions/` | Live/eval session JSONL + crops |
| `outputs/har_analysis/` | Notebook 06 charts |
| `outputs/ensemble_avance5/` | Avance 5 comparison table |
| `checkpoints/` | All `.pt` checkpoints (v1 + crop-aligned) |
