# Model weights (local install)

Large ONNX files are **not in git** (see root `.gitignore`). Everyone clones the repo, then downloads weights once on their machine.

## Automatic (recommended)

From the repo root:

```bash
./run-local.sh
```

On the **first run**, if face models are missing, this calls `./models/install_face_models.sh` automatically (~40 MB download from Hugging Face). Later runs skip download if files already exist.

## Manual install

```bash
./models/install_face_models.sh
```

Requires **internet** and one of:

- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli): `curl -LsSf https://hf.co/cli/install.sh | bash`
- Or **uv** / **Python** with `huggingface-hub` (the install script tries both)

Optional faster downloads: `brew install git-xet && git xet install`

## What gets created locally (gitignored)

| Folder | Main file | Size (approx.) |
|--------|-----------|----------------|
| `models/face_detection_yunet/` | `face_detection_yunet_2023mar.onnx` | ~0.2 MB |
| `models/face_recognition_sface/` | `face_recognition_sface_2021dec.onnx` | ~39 MB |

Sources:

- [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet)
- [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface)

Used by **vision-ops-backend** for live face detection and identity on `/live`.

## What stays in git

| Path | Purpose |
|------|---------|
| `models/README.md` | This file |
| `models/install_face_models.sh` | Download script |
| `models/dinov3-main/` | DINOv3 reference code (I+D) |
| `models/vjepa2-main/` | V-JEPA reference code (I+D) |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Face models missing` on Live | Run `./models/install_face_models.sh` |
| `hf: command not found` | Install HF CLI or use `uv` (install script fallback) |
| Download slow / fails | Check network; retry install script |

Personal enrollment data (your face embedding) is separate: `vision-ops-backend/data/faces/` (also gitignored).
