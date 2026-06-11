# notebooks_v2 — Improved HAR Pipeline

Full end-to-end pipeline targeting **90%+ subject-split accuracy** on the InHARD 12-class dataset.

## Quick Start

```bash
cd notebooks_v2
# Run full pipeline (all improvements active by default)
jupyter notebook 00_Pipeline_Run_All.ipynb
```

Or run step-by-step:

| Notebook | Purpose |
|---|---|
| `00_Pipeline_Run_All.ipynb` | Master: runs all steps in one cell |
| `01_Data_and_Strategy.ipynb` | Inventory, class balance, 3-view extraction |
| `02_Embedding_Extraction.ipynb` | V-JEPA2 + DINOv2 with attention pooling |
| `03_Train_HAR_Head.ipynb` | MLP/GRU head with Focal Loss + Mixup + SupCon |
| `04_Analysis_and_Visualization.ipynb` | All 13 charts incl. UMAP, frame strips, YOLO grid |
| `05_Mock_Video_Eval.ipynb` | Annotated render on Industrial-One + madera videos |

## Improvements vs v1

| Feature | v1 | v2 |
|---|---|---|
| Training data | 100 clips/class cap | **All available clips** |
| Frame extraction | YOLO on full 1280×720 mosaic | **Top-down sub-view + YOLO** |
| Temporal aggregation | Mean-pool CLS tokens | **Attention pooling** |
| Loss function | CrossEntropy + class weights | **Focal Loss (γ=2)** |
| Batch sampling | Uniform random | **WeightedRandomSampler** |
| Augmentation | None | **Mixup (3× virtual samples)** |
| Training epochs | 25 | **100 + CosineAnnealingLR** |
| Optional SupCon | ✗ | **✓ (Supervised Contrastive)** |
| Optional GRU head | ✗ | **✓ (Bi-GRU sequential)** |
| Analysis charts | 8 charts | **13 charts + UMAP + frame strips** |

## Key Files

```
notebooks_v2/
├── lib/
│   ├── constants.py         # All tuneable knobs
│   ├── crop_extract.py      # 3-view extraction + YOLO crops
│   ├── dino_embeddings.py   # DINOv2 with temporal attention pooling
│   ├── har_model.py         # HarMLP, HarGRU, SupConProjector
│   ├── har_train.py         # FocalLoss, SupConLoss, Mixup, WeightedSampler
│   ├── har_analysis.py      # All 13 visualization functions
│   └── pipeline.py          # End-to-end orchestrator
├── checkpoints/             # Saved model weights
└── outputs/                 # Embeddings, metrics, charts
```

## Expected Accuracy Trajectory

| Stage | Accuracy (subject-split) |
|---|---|
| v1 baseline | 30–35% |
| + all clips + 3-view + epochs | ~55–65% |
| + Focal + Mixup + Sampler | ~70–80% |
| + SupCon + GRU head | **85–90%+** |

See [`improvements.md`](../improvements.md) for full analysis.
