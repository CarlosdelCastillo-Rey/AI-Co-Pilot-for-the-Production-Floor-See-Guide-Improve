# VisionOps — Fase 0 pipeline (notebooks)

Scripts atómicos del plan VisionOps: cada etapa es un **notebook Jupyter** ejecutable de forma aislada, con salidas bajo `outputs/` (ignorado por git).

**Nota:** Los notebooks `05_semantic_event` y `07_telegram_webhook` son legado del plan inicial y **no forman parte del demo VisionOps** (`./run-local.sh`). El stack productivo usa Strands/Ollama + MailerSend en `vision-ops-alerting/`.

## Orden de ejecución (recomendado)

```text
01_capture_stream.ipynb
    → 02_segment_frames.ipynb
        → 03_detect_and_track.ipynb
            → 04_build_event_buffer.ipynb
                ├→ 06_generate_heatmap.ipynb
                └→ 08_vjepa_action_probe.ipynb   (opcional; pesos HF/GPU)
```

## Prerrequisitos

```bash
# Desde la raíz del repositorio
uv sync --all-groups
cp .env.example .env   # opcional: REDIS_URL para notebook 04
```

Abrir notebooks con kernel **Python 3.11** del `.venv`:

```bash
uv run jupyter lab scripts/
```

Los notebooks importan utilidades desde [`_common/io.py`](_common/io.py). La primera celda de código llama a `ensure_scripts_on_path()`.

## Tabla entrada / salida

| Notebook | Entrada principal | Salida en `outputs/` |
|----------|-------------------|----------------------|
| 01 | Webcam, `.mp4`, RTSP o InHARD (`SOURCE=auto`) | `01_capture/` frames + `metadata.json` |
| 02 | `01_capture/` o video | `02_segments/` frames, `clips/`, `manifest.json` |
| 03 | Clip de `02` | `03_track/tracks.jsonl`, `annotated.mp4` |
| 04 | `tracks.jsonl` | `04_events/events.jsonl` (+ Redis opcional) |
| 06 | `tracks.jsonl` | `06_heatmap/heatmap.png`, `heatmap.json` |
| 08 | Clip de `02` | `08_vjepa/embedding.npy`, `probe_result.json` |

## Variables de entorno

Ver [`.env.example`](../.env.example) en la raíz:

| Variable | Notebook |
|----------|----------|
| `REDIS_URL` | 04 (opcional) |

## Datos InHARD

Clips de prueba: `data_sample/InHARD-master/01-InHARD/Segmented/RGBSegmented/` (no versionados; extraer desde [Zenodo 4003541](https://zenodo.org/record/4003541)).

Con `SOURCE=auto`, el notebook **01** usa el primer `.mp4` encontrado bajo esa ruta; si no hay video local, intenta webcam `0`.

Documentación del dataset: [`data_sample/InHARD-master/analysis.md`](../data_sample/InHARD-master/analysis.md).

## Ejecución por línea de comandos (smoke test)

```bash
uv sync --all-groups
for nb in scripts/0{1,2,3,4}_*.ipynb; do
  uv run jupyter nbconvert --execute "$nb" --to notebook --inplace
done
```

El **08** puede `SKIPPED` sin pesos V-JEPA / GPU.

## Criterios de éxito (plan Fase 0)

- **01:** ≥30 frames en `outputs/01_capture/` (o ≥1 si solo hay stream corto de prueba).
- **03:** `tracks.jsonl` no vacío; varios `track_id` en clip corto.
- **04:** al menos un evento en `events.jsonl`.
- Evidencia recomendada: 1 run completo `01→04` sobre clip InHARD o video local.

## Smoke test (última verificación)

Ejecutado con clip InHARD (`SOURCE=auto`) vía `jupyter nbconvert --execute`:

| Paso | Resultado (referencia local) |
|------|------------------------------|
| `01` | 11+ frames en `outputs/01_capture/`, `metadata.json` |
| `02` | `manifest.json`, clips bajo `outputs/02_segments/clips/` |
| `03` | `tracks.jsonl` (~200+ líneas), `annotated.mp4` |
| `04` | `events.jsonl` (warning / idle / forklift_zone) |
| `06` | `heatmap.png` + `heatmap.json` |
| `08` | `embedding.npy` + `probe_result.json` (`SKIPPED` sin `transformers`/pesos HF) |

Para V-JEPA completo: instalar `transformers` y descargar pesos según `models/vjepa2-main/README.md`.

_Regenerar notebooks desde plantilla:_ `uv run python scripts/_generate_notebooks.py`
