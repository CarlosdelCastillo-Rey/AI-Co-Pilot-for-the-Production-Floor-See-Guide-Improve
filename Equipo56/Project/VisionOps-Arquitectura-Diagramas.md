# VisionOps — Diagramas de arquitectura (Equipo 56)

Documento de referencia con los diagramas acordados para el **AI Co-Pilot / VisionOps**: backend local (`vision-ops-backend`), frontend (`vision-ops-app`), modelos en `models/`, y pipeline Fase 0 en `scripts/`.

**Relacionado:** [Project-context.md](./Project-context.md) · [Visión por Computador Industrial](./Visión%20por%20Computador%20Industrial:%20IA%20Co-Piloto.md) · [PRD.pdf](./PRD.pdf) · [Project-links.md](./Project-links.md)

**Última actualización:** mayo 2026

---

## 1. Vista general: quién posee la cámara

El navegador **no envía frames** al API en el flujo live actual. El backend abre la webcam (OpenCV), procesa en un hilo y **empuja** JPEGs al cliente.

```mermaid
flowchart LR
  subgraph browser [vision-ops-app]
    IMG["img src=stream MJPEG"]
    REST[fetch JSON APIs]
  end
  subgraph backend [vision-ops-backend]
    WC[OpenCV webcam thread]
    FE[SFace + YuNet]
    WC --> FE
    FE --> JPEG[JPEG con cajas dibujadas]
  end
  JPEG -->|multipart/x-mixed-replace| IMG
  REST -->|POST name only| backend
```

| Pieza | Entrada | Salida |
|-------|---------|--------|
| `GET /api/cameras` | — | JSON: `status`, `streamUrl`, `overlays`, `error` |
| `GET /api/cameras/webcam-0/stream` | — | MJPEG continuo (cajas ya pintadas en servidor) |
| `POST /api/faces/enroll` | `{ "name": "..." }` | JSON; el servidor muestrea su propio buffer de webcam |

---

## 2. Flujo de datos Live (Camera 01 — webcam)

```mermaid
sequenceDiagram
  participant B as Browser /live
  participant API as FastAPI
  participant T as WebcamCapture thread
  participant F as SFaceLiveEngine

  B->>API: GET /api/cameras
  API->>T: is_running, overlays
  API-->>B: streamUrl + status live

  B->>API: GET /api/cameras/webcam-0/stream
  loop cada ~12 fps
    T->>T: read frame
    T->>F: process (detect cada N frames)
    F-->>T: frame anotado + overlays metadata
    T->>T: imencode JPEG
    API-->>B: parte MJPEG
  end

  B->>API: POST /api/faces/enroll {name}
  API->>T: get_latest_frame x12
  API->>F: enroll_from_frames
  API-->>B: owner.npz guardado
```

**Almacenamiento local (gitignored):**

- `vision-ops-backend/data/faces/owner.npz` — embedding + nombre
- `vision-ops-backend/data/faces/enrollment_preview.jpg` — referencia visual (no usada para match)
- Video live: solo RAM

---

## 3. Arranque local y modelos ONNX (cara)

```mermaid
flowchart TD
  A[./run-local.sh] --> B{¿Existen ONNX YuNet + SFace?}
  B -->|No| C[models/install_face_models.sh]
  C --> D[~40 MB Hugging Face]
  B -->|Sí| E[uv sync backend]
  D --> E
  E --> F[npm install frontend]
  F --> G[Backend :8000 + Frontend :3000]
  G --> H[/live y /identity]
```

**En git:** `models/install_face_models.sh`, `models/README.md`  
**No en git:** `*.onnx`, carpetas `face_detection_yunet/`, `face_recognition_sface/`

---

## 4. Cámara Android como fuente (Wi‑Fi — diseño)

Hoy solo `CAMERA_INDEX` (webcam Mac). Opción documentada: app **IP Webcam** en Android → URL HTTP → OpenCV.

```mermaid
flowchart LR
  Phone[Android IP Webcam]
  Mac[vision-ops-backend]
  Browser[Browser /live]
  Phone -->|HTTP MJPEG WiFi| Mac
  Mac -->|MJPEG anotado| Browser
```

| Método | Estado |
|--------|--------|
| Webcam Mac (`CAMERA_INDEX=0`) | Implementado |
| `CAMERA_SOURCE_URL=http://IP:8080/video` | Propuesto (env) |
| DroidCam como cámara virtual | Probar índices 1, 2, 3 |
| Browser en teléfono sube frames | No implementado (nuevo API) |

---

## 5. Dos pipelines: Camera 01 vs Camera 02+

```mermaid
flowchart TB
  subgraph cam1 [Camera 01 - webcam-0]
    W[Webcam thread]
    S[SFace + YuNet ONNX]
    W --> S --> MJPEG[GET .../stream]
  end
  subgraph cam2 [Camera 02+ - mock / InHARD / RTSP futuro]
    C[Clip InHARD o frames outputs/02]
    D[DINOv3 - espacial]
    V[V-JEPA 2 - temporal]
    C --> D
    C --> V
    D --> VAPI["/api/vision/* propuesto"]
    V --> VAPI
  end
  UI[vision-ops-app /live]
  MJPEG --> UI
  VAPI --> UI
```

| Cámara | Modelo | Modo | API |
|--------|--------|------|-----|
| 01 Webcam | YuNet + SFace | Tiempo real (&lt;100 ms objetivo edge) | `/api/faces/*`, `/api/cameras/.../stream` |
| 02 Warehouse (mock) | V-JEPA | Clip / batch | `/api/vision/*` (propuesto) |
| 01 Assembly (mock) | DINOv3 | Frame / batch | `/api/vision/*` (propuesto) |

---

## 6. API espejo: `/api/faces` vs `/api/vision` (implementado)

```mermaid
flowchart LR
  subgraph faces [/api/faces - implementado]
    F1[GET status]
    F2[GET storage]
    F3[POST enroll name]
    F4[GET preview JPG]
    F5[DELETE enroll]
  end
  subgraph vision [/api/vision - implementado]
    V1[GET status]
    V2[GET storage]
    V3[POST probe camera_id mode]
    V4[GET artifacts heatmap json]
    V5[overlays en GET /api/cameras]
  end
```

---

## 7. Sinergia DINOv3 + V-JEPA (marco académico)

Alineado a [Visión por Computador Industrial](./Visión%20por%20Computador%20Industrial:%20IA%20Co-Piloto.md) y README del repo.

```mermaid
flowchart LR
  F[Frame o clip video]
  F --> D[DINOv3<br/>segmentación / features densas]
  F --> V[V-JEPA 2<br/>embedding temporal]
  D --> R[ROI / zonas / heatmap]
  V --> A[acción / anomalía / anticipación]
  R --> FUS[Fusión latente]
  A --> FUS
  FUS --> E[Eventos JSON / bitácora]
  E --> NLP[NLP Co-Pilot futuro]
```

| Modelo | Carpeta repo | Entrada | Uso en pruebas |
|--------|--------------|---------|----------------|
| DINOv3 | `models/dinov3-main/` | Frame | Heatmap, drift escena, ROI para clips |
| V-JEPA 2 | `models/vjepa2-main/` | Clip 16–64 frames | Embedding, anomalía, acciones InHARD |
| SFace/YuNet | `models/face_*` | Frame live | Identidad operador (Camera 01) |

---

## 8. Pipeline Fase 0 (`scripts/` — notebooks)

```mermaid
flowchart TD
  N01[01_capture_stream]
  N02[02_segment_frames]
  N03[03_detect_and_track]
  N04[04_build_event_buffer]
  N05[05_semantic_event]
  N06[06_generate_heatmap]
  N07[07_telegram_webhook]
  N08[08_vjepa_action_probe]

  N01 --> N02 --> N03 --> N04
  N04 --> N05
  N04 --> N06
  N04 --> N07
  N02 --> N08
```

| Notebook | Salida principal |
|----------|------------------|
| 01 | `outputs/01_capture/` |
| 02 | `outputs/02_segments/clips/` |
| 03 | `outputs/03_track/tracks.jsonl` |
| 04 | `outputs/04_events/events.jsonl` |
| 08 | `outputs/08_vjepa/embedding.npy`, `probe_result.json` |

---

## 9. PRD vs implementación actual (decisión de stack)

```mermaid
flowchart TB
  subgraph live [Live MVP - en curso]
    RT[RTSP futuro]
    CV[OpenCV ingest]
    YO[YOLO + DeepSORT plan PRD]
    SF[SFace face ID - hecho]
    CV --> SF
    RT -.-> CV
  end
  subgraph batch [Post-turno / I+D]
    IN[InHARD clips]
    DI[DINOv3]
    VJ[V-JEPA]
    GM[Gemini bitácora]
    IN --> DI
    IN --> VJ
    VJ --> GM
  end
  UI[vision-ops-app REQ-01 a REQ-04]
  live --> UI
  batch --> UI
```

**Decisión documentada en plan:** YOLO+DeepSORT para live; DINOv3/V-JEPA en clips (no bloquean MVP PRD).

---

## 10. Roadmap de pruebas en cámara mock (`cam-02`)

```mermaid
flowchart LR
  T0[T0 API /api/vision status]
  T1[T1 POST probe V-JEPA InHARD]
  T2[T2 DINO heatmap cam-01]
  T3[T3 overlays + eventos Live]
  T4[T4 RTSP / teléfono IP]
  T0 --> T1 --> T2 --> T3 --> T4
```

| Fase | Entregable |
|------|------------|
| T0 | Router `vision.py` + storage paths |
| T1 | Probe sobre clip InHARD → JSON anomalía |
| T2 | Heatmap DINO en `cam-01` |
| T3 | `GET /api/cameras` con `overlays` reales |
| T4 | `CAMERA_SOURCE_URL` o RTSP |

---

## 11. Ejemplo de probe (cuerpo API propuesto)

```mermaid
sequenceDiagram
  participant UI as Vision Lab UI
  participant API as POST /api/vision/probe
  participant Job as Probe job
  participant Out as outputs/08_vjepa

  UI->>API: camera_id cam-02, mode vjepa_anomaly
  API->>Job: clip InHARD path
  Job->>Out: embedding.npy, probe_result.json
  Job-->>API: event severity + scores
  API-->>UI: JSON + artifact URLs
  UI->>API: GET /api/cameras
  API-->>UI: cam-02 overlays actualizados
```

---

## 12. Mapa de rutas del repositorio

```mermaid
flowchart TB
  ROOT[Repo raíz]
  ROOT --> RL[run-local.sh]
  ROOT --> BE[vision-ops-backend]
  ROOT --> FE[vision-ops-app]
  ROOT --> SCR[scripts/ Fase 0]
  ROOT --> MOD[models/]
  ROOT --> EQ[Equipo56/]

  BE --> API[src/vision_ops_backend]
  API --> R1[routers/cameras.py]
  API --> R2[routers/faces.py]
  API --> R3[routers/vision.py futuro]

  MOD --> M1[face_* ONNX]
  MOD --> M2[dinov3-main]
  MOD --> M3[vjepa2-main]

  FE --> P1["/live"]
  FE --> P2["/identity"]
  FE --> P3["/vision-lab futuro"]
```

---

## 13. URLs locales (demo)

| Servicio | URL |
|----------|-----|
| Live UI | http://localhost:3000/live |
| Identidad facial | http://localhost:3000/identity |
| API health | http://localhost:8000/health |
| Proxy Next → API | `/vision-api/*` → `:8000` |

---

## Cómo ver los diagramas

- **GitHub / GitLab:** renderizado Mermaid nativo en `.md`
- **VS Code / Cursor:** preview Markdown con extensión Mermaid
- **Exportar PNG:** [mermaid.live](https://mermaid.live) — pegar bloque `mermaid`

---

_Equipo 56 — VisionOps. Complementa el PRD y el plan en `.cursor/plans/`._
