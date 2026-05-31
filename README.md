# AI Co-Pilot para el Piso de Producción: Ver, Guiar, Mejorar

> Sistema de visión multi-cámara orientado a Industria 4.0 que convierte cámaras IP existentes en un **gemelo digital operativo** del piso de producción — asistiendo a supervisores de planta sin sustituirlos.

**Titular:** Alignity IQ Edge, LLC — Houston, Texas, EUA  
**Equipo #56 (MNA-V · Tec de Monterrey):** Landy Haydee Schlebach Osorio · Carlos Pano Hernández · Carlos Fernando Del Castillo Rey  
**Asesor académico:** Dr. Gerardo Camacho  
**Patrocinador industrial:** Dr. José Jacobo Eluani Vázquez (Representante Legal, Alignity IQ Edge, LLC)

---

## Resumen ejecutivo

En planta, la supervisión suele depender de recorridos físicos o de mirar monitores de forma pasiva. Este proyecto plantea un entorno **VisionOps**: ingestión de video multi-cámara, extracción de señales visuales con **modelos fundacionales** (DINOv3, V-JEPA 2.x) y orquestación mediante **NLP** para alertas, bitácoras y reportes accionables.

| Pilar | Qué aporta | Stack demo actual |
|-------|------------|-------------------|
| **Ver** | Ingesta multi-cámara y comprensión de escena/acción | Webcam MJPEG + mocks industriales + probes DINO/V-JEPA + SFace |
| **Guiar** | Alertas en baja latencia + workflow de turno | Reglas SQLite, clasificador Strands/Ollama, email MailerSend, ack/resolve en timeline |
| **Mejorar** | Bitácora visual, KPIs post-turno, asesor IA | Timeline industrial, OEE/CoQ/Pareto dinámicos, plant settings, **VisionOps AI Advisor** (chat) |

**Novedades recientes del demo:** login email/contraseña, atribución por usuario en timeline y alertas, página de **Settings** (costos y fórmulas KPI), analytics industrial (OEE, CoQ, Pareto), workflow de eventos (ack → resolve + códigos de razón), campana de notificaciones, bot flotante **VisionOps AI Advisor** (Strands + Ollama), servidor MCP opcional para contexto SQLite.

Documentación académica extendida: `Equipo56/Project/Project-context.md`, `Equipo56/Project/Visión por Computador Industrial: IA Co-Piloto.md`.

---

## VisionOps demo stack (for developers & AI agents)

The runnable demo consists of **three services** started together via `./run-local.sh`:

| Service | Directory | Port | Role |
|---------|-----------|------|------|
| **vision-ops-backend** | `vision-ops-backend/` | **8000** | Webcam MJPEG, SFace face ID, DINO/V-JEPA vision probes |
| **vision-ops-alerting** | `vision-ops-alerting/` | **8001** | SQLite persistence, alert rules, timeline, analytics, email |
| **vision-ops-app** | `vision-ops-app/` | **3000** | Next.js 16 dashboard — Live, Timeline, Analytics, Alerts, Settings, Identity, Vision Lab, login, AI advisor |

### System architecture

```mermaid
flowchart TB
    subgraph Browser["Browser (localhost:3000)"]
        UI["vision-ops-app<br/>Next.js App Router"]
    end

    subgraph Proxies["Next.js rewrites"]
        VP["/vision-api/*"]
        AP["/alerting-api/*"]
    end

    subgraph Backend8000["vision-ops-backend :8000"]
        CAM["/api/cameras<br/>MJPEG + mocks"]
        FACE["/api/faces<br/>SFace enrollment"]
        VIS["/api/vision<br/>DINO + V-JEPA probes"]
    end

    subgraph Alerting8001["vision-ops-alerting :8001"]
        AUTH["/api/auth/*"]
        RULES["/api/alerts/*"]
        TL["/api/timeline/*<br/>ack · resolve"]
        AN["/api/analytics/*<br/>OEE · CoQ · Pareto"]
        SET["/api/settings/*"]
        ADV["/api/advisor/*<br/>VisionOps AI chat"]
        DB_CAM["/api/cameras<br/>DB + merge runtime"]
        AGENT["Strands agents<br/>Ollama classify + advisor"]
        MS["MailerSend email"]
        SQLITE[("SQLite<br/>vision_ops.db")]
    end

    UI --> VP
    UI --> AP
    VP --> Backend8000
    AP --> Alerting8001
    DB_CAM -->|"GET /api/cameras"| Backend8000
    Alerting8001 --> SQLITE
    AGENT --> MS
    RULES --> AGENT
```

### Alert pipeline (email → timeline)

```mermaid
sequenceDiagram
    participant V as Vision / manual curl
    participant A as vision-ops-alerting
    participant O as Ollama (optional)
    participant M as MailerSend
    participant DB as SQLite

    V->>A: POST /api/alerting/email<br/>IndustrialContext JSON
    A->>O: classify_case (Strands) or fallback rules
    O-->>A: case_type + severity
    A->>DB: require enabled alert_rule for case_type
    A->>M: render template + send (unless dry_run)
    A->>DB: INSERT events + alert_deliveries
    A-->>V: event_id, delivery_id
    Note over A,DB: Event appears on /timeline UI
```

### Live camera data flow

```mermaid
flowchart LR
    subgraph Alerting["vision-ops-alerting"]
        DB[(cameras table)]
        MERGE["list_cameras_merged()"]
    end

    subgraph Vision["vision-ops-backend"]
        WEB["webcam-0 MJPEG"]
        MOCK["cam-01 Assembly<br/>cam-02 Warehouse"]
    end

    DB --> MERGE
    MERGE -->|"httpx GET /api/cameras"| Vision
    WEB --> MERGE
    MOCK --> MERGE
    MERGE -->|"streamUrl, overlays,<br/>heatmapUrl, visionProbe"| APP["/live UI"]
```

**Key design choice:** Camera **metadata** lives in alerting SQLite; **runtime** (streams, probe artifacts) comes from vision-ops-backend. The alerting service merges both on `GET /api/cameras`.

### Python environments (three packages)

| Directory | Tool | Purpose |
|-----------|------|---------|
| Repository root | `pyproject.toml` + `uv sync --all-groups` | Notebooks, YOLO/DeepSORT research |
| `vision-ops-backend/` | `uv sync` | FastAPI webcam + face + vision probes |
| `vision-ops-alerting/` | `uv sync` (+ `--extra mcp` for MCP server) | FastAPI alerts, SQLite, Strands agents, JWT auth |
| `vision-ops-app/` | `npm install` | Next.js frontend |

Root `requirements.txt` is a legacy pip pin list for notebooks; prefer **`uv sync`** at root or in each service directory.

---

## Repository structure

```text
.
├── run-local.sh                 # Starts all 3 services (recommended)
├── vision-ops-backend/          # FastAPI — webcam, faces, vision probes (:8000)
│   ├── src/vision_ops_backend/
│   │   ├── main.py
│   │   ├── routers/             # cameras, faces, vision, health
│   │   ├── face/                # YuNet + SFace ONNX live recognition
│   │   └── vision/              # DINO heatmap + V-JEPA anomaly probes
│   ├── data/faces/              # gitignored — enrollment embeddings
│   └── data/vision/             # gitignored — probe artifacts
├── vision-ops-alerting/         # FastAPI — rules, timeline, email, auth, advisor (:8001)
│   ├── src/vision_ops_alerting/
│   │   ├── main.py
│   │   ├── agent.py             # Strands alert classifier + MailerSend
│   │   ├── advisor_agent.py     # Strands floor advisor (chat)
│   │   ├── auth_deps.py         # JWT Bearer dependency
│   │   ├── db/models.py         # SQLAlchemy models (source of truth)
│   │   ├── routers/             # auth, alerts, timeline, analytics, settings, advisor, …
│   │   └── services/            # events, workflow, industrial_analytics, plant_settings
│   ├── mcp/db_context_server.py # Optional MCP — SQLite ops context for Cursor
│   ├── docs/schema.sql          # Reference DDL (may lag models.py)
│   └── data/vision_ops.db       # gitignored — auto-created SQLite
├── vision-ops-app/              # Next.js 16 + React 19 + Tailwind 4 (:3000)
│   ├── app/(dashboard)/         # live, timeline, analytics, alerts, settings, identity, vision-lab
│   ├── app/login/               # Email/password sign-in + register
│   ├── components/advisor/      # VisionOps AI floating chat (all dashboard pages)
│   ├── components/auth/         # AuthProvider session gate
│   ├── lib/api.ts               # All fetch helpers + proxy URL logic + JWT headers
│   └── next.config.ts           # /vision-api and /alerting-api rewrites
├── models/                      # DINOv3, V-JEPA reference code + face ONNX installer
├── Equipo56/Project/            # Academic context (Markdown)
├── notebooks/                   # ML experimentation
├── data_sample/InHARD-master/   # Industrial action dataset reference
├── pyproject.toml               # Root Python deps (research / notebooks)
└── README.md
```

---

## Prerequisites

| Tool | Version | Used by |
|------|---------|---------|
| **Python** | 3.11–3.12 | Both FastAPI backends (`requires-python >=3.11,<3.13`) |
| **[uv](https://docs.astral.sh/uv/)** | latest | Python dependency management |
| **Node.js + npm** | 20+ recommended | `vision-ops-app` |
| **macOS camera access** | — | Webcam stream + face enrollment (Terminal/IDE permission) |
| **Ollama** (optional) | — | LLM for alert classification **and** VisionOps AI Advisor; rule-based fallbacks work without it |
| **Internet** (first run) | — | Face ONNX download (~40 MB), npm packages |

### One-time setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <url-del-repositorio>
cd AI-Co-Pilot-for-the-Production-Floor-See-Guide-Improve

# Root research environment (notebooks, optional)
uv sync --all-groups

# Face models (required for Identity page — not in git)
./models/install_face_models.sh

# Env files (run-local.sh copies from .example if missing)
cp vision-ops-app/.env.local.example vision-ops-app/.env.local
cp vision-ops-alerting/.env.example vision-ops-alerting/.env
cp vision-ops-backend/.env.example vision-ops-backend/.env   # optional
```

---

## Quick start (all services)

From the **repository root**:

```bash
./run-local.sh
```

This script:

1. Ensures face ONNX models exist (`models/install_face_models.sh`)
2. Runs `uv sync` in backend + alerting
3. Runs `npm install` in frontend if needed
4. Frees ports 8000, 8001, 3000
5. Starts all three servers (Ctrl+C stops everything and releases the webcam)

| URL | Page |
|-----|------|
| http://localhost:3000/login | Sign in / create account (required before dashboard) |
| http://localhost:3000/live | Multi-camera grid + live stats |
| http://localhost:3000/timeline | Post-shift log — ack / resolve workflow |
| http://localhost:3000/analytics | OEE, CoQ, Pareto, heatmap, KPI tooltips |
| http://localhost:3000/alerts | Alert rules + email templates CRUD |
| http://localhost:3000/settings | Plant cost variables + KPI formula reference |
| http://localhost:3000/identity | Face enrollment (SFace) |
| http://localhost:3000/vision-lab | DINO / V-JEPA probes |
| http://localhost:8000/health | Backend liveness |
| http://localhost:8001/health | Alerting liveness |
| http://localhost:8001/docs | Alerting OpenAPI (auth, advisor, timeline, …) |

**Default login** (seeded on first start when `users` is empty): `admin@visionops.local` / `admin123`

**VisionOps AI Advisor:** blue floating button (bottom-right) on every dashboard page — context-aware chat powered by Strands + Ollama.

### Run services individually

```bash
# Backend only
cd vision-ops-backend && uv sync
uv run uvicorn vision_ops_backend.main:app --reload --host 0.0.0.0 --port 8000

# Alerting only
cd vision-ops-alerting && uv sync
uv run uvicorn vision_ops_alerting.main:app --reload --host 0.0.0.0 --port 8001

# Frontend only
cd vision-ops-app && npm install && npm run dev
```

---

## Environment variables

### vision-ops-app (`vision-ops-app/.env.local`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | vision-ops-backend (rewrites + SSR) |
| `NEXT_PUBLIC_ALERTING_URL` | `http://localhost:8001` | vision-ops-alerting (SSR direct fetch) |

Browser calls use proxies defined in `next.config.ts`:

- `/vision-api/*` → backend `:8000`
- `/alerting-api/*` → alerting `:8001`

### vision-ops-backend (`vision-ops-backend/.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Must include your UI origin |
| `PUBLIC_API_BASE` | `http://localhost:8000` | Absolute stream URLs in JSON |
| `MJPEG_FPS` | `12` | Webcam stream frame rate |
| `FACE_ENABLED` | `true` | SFace overlay on MJPEG |
| `OWNER_NAME` | `You` | Default display name |
| `VISION_ENABLED` | `true` | Include cam-01/cam-02 mock cameras |

### vision-ops-alerting (`vision-ops-alerting/.env`)

All vars use prefix `ALERTING_` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALERTING_DRY_RUN` | `true` | Safe default — no real email unless `false` |
| `ALERTING_MAILERSEND_API_TOKEN` | — | Required for real sends |
| `ALERTING_FROM_EMAIL` / `ALERTING_TO_EMAIL` | — | Sender + comma-separated recipients |
| `ALERTING_OLLAMA_MODEL` | `llama3.1` | Strands classifier model |
| `ALERTING_CORS_ORIGINS` | `http://localhost:3000` | CORS |
| `ALERTING_AUTH_SECRET` | dev default | JWT signing — **change in production** |
| `ALERTING_AUTH_TOKEN_HOURS` | `72` | Session token lifetime |
| `ALERTING_SEED_ADMIN_EMAIL` | `admin@visionops.local` | First-user seed when `users` table is empty |
| `ALERTING_SEED_ADMIN_PASSWORD` | `admin123` | Default admin password (dev only) |
| `ALERTING_SEED_ADMIN_NAME` | `Plant Supervisor` | Display name on workflow actions |
| `ALERTING_SEED_DB` | `false` | If `true`, inserts demo rules/events on empty DB |
| `ALERTING_DATABASE_URL` | `sqlite:///.../data/vision_ops.db` | SQLite path (use absolute path in DBeaver) |
| `ALERTING_VISION_BACKEND_URL` | `http://localhost:8000` | Health/latency probes |

### Root `.env` (optional — notebooks only)

Not used by `./run-local.sh`. Copy from `.env.example` only if you run root notebook experiments.

| Variable | Purpose |
|----------|---------|
| `REDIS_URL` | Optional event buffer for notebook 04 (`redis://localhost:6379/0`) |

Demo auth uses `ALERTING_AUTH_SECRET` in `vision-ops-alerting/.env`, not the root `.env`.

---

## Authentication

Simple **email + password** login (no OAuth). Implemented in **vision-ops-alerting**; the Next.js app stores a JWT in `localStorage` and sends `Authorization: Bearer …` on mutating API calls.

| Item | Detail |
|------|--------|
| UI | `/login` — sign in or create account (name, email, password) |
| API | `POST /api/auth/login`, `POST /api/auth/register`, `GET /api/auth/me` |
| Storage | SQLite table `users` (`id`, `email`, `name`, `password_hash`, `created_at`) |
| Roles | **Not implemented** — all authenticated users share the same permissions |
| Attribution | User **display name** stored on `events.acknowledged_by`, `events.resolved_by`, `alert_rules.updated_by`, `plant_config.updated_by` |

Protected routes (require Bearer token): timeline ack/resolve, alert rule CRUD, email template CRUD, plant settings PATCH.

```bash
# Login (returns token + user)
curl -s -X POST http://localhost:8001/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@visionops.local","password":"admin123"}' | jq .

# Use token on protected calls
TOKEN="<paste token>"
curl -s -X PATCH http://localhost:8001/api/timeline/evt-xxx/acknowledge \
  -H "Authorization: Bearer $TOKEN"
```

---

## VisionOps AI Advisor (chat bot)

Floating **ops advisor** on every dashboard page (`components/advisor/VisionOpsAdvisor.tsx`). Uses the same **Strands + Ollama** stack as alert classification, with tools that query live SQLite context (cameras, open events, rules, KPIs).

| Endpoint | Purpose |
|----------|---------|
| `GET /api/advisor/welcome?page=live` | Intro when chat opens (page-aware) |
| `POST /api/advisor/chat` | User message → advisor reply + snapshot |
| `GET /api/advisor/context` | Raw operational snapshot (debug / MCP parity) |

Requires **Ollama** with `ALERTING_OLLAMA_MODEL` (default `llama3.1`) for full replies; deterministic fallbacks exist for greetings.

### Optional MCP server (Cursor / agents)

`vision-ops-alerting/mcp/db_context_server.py` exposes the same DB context over MCP stdio:

```bash
cd vision-ops-alerting
uv sync --extra mcp
uv run python mcp/db_context_server.py
```

See file header for Cursor `mcpServers` JSON config.

---

## Database schema (vision-ops-alerting)

**File:** `vision-ops-alerting/data/vision_ops.db` (gitignored, WAL mode, auto-created on startup)

**Source of truth:** `vision-ops-alerting/src/vision_ops_alerting/db/models.py`  
**Reference DDL:** `vision-ops-alerting/docs/schema.sql`

### Entity relationship

```mermaid
erDiagram
    alert_rules ||--o{ events : triggers
    events ||--o{ alert_deliveries : notifies
    alert_rules }o--o| email_templates : uses
    cameras ||--o{ events : "camera_id (logical)"
    analytics_daily }o--|| cameras : "optional camera_id"
    analytics_heatmaps }o--|| cameras : camera_id

    alert_rules {
        text id PK
        text case_type
        text severity
        bool enabled
        bool notify_email
    }
    email_templates {
        text id PK
        text case_type
        text subject
        bool is_builtin
    }
    events {
        text id PK
        text rule_id FK
        text camera_id
        text case_type
        text severity
        text resolution_status
        text acknowledged_by
        text resolved_by
        datetime occurred_at
    }
    users {
        text id PK
        text email UK
        text name
        text password_hash
    }
    plant_config {
        text id PK
        float line_cost_per_minute
        text updated_by
    }
    industrial_reason_codes {
        text code PK
        text label
    }
    alert_deliveries {
        text id PK
        text event_id FK
        text channel
        text status
    }
    cameras {
        text id PK
        text name
        text source_type
        text backend_camera_id
        int sort_order
    }
    analytics_daily {
        text id PK
        text event_date
        text shift
        float uptime_pct
    }
    analytics_heatmaps {
        text id PK
        text camera_id
        text grid_json
    }
    health_metric_samples {
        text id PK
        text service
        float latency_ms
    }
```

### Tables summary

| Table | Purpose |
|-------|---------|
| `users` | App login accounts (bcrypt passwords) |
| `cameras` | Camera registry for Live UI; links to backend via `backend_camera_id` (e.g. `webcam-0`, `cam-01`) |
| `alert_rules` | Configurable rules on `/alerts`; one enabled rule per `case_type` required for email dispatch |
| `email_templates` | Built-in + cloned HTML email templates per case type |
| `events` | Timeline incidents — includes workflow: `resolution_status`, `acknowledged_by/at`, `resolved_by/at`, reason codes, downtime/scrap |
| `alert_deliveries` | Email send log linked to events |
| `industrial_reason_codes` | Close-out reason codes for timeline resolve modal |
| `plant_config` | Editable plant costs, shift hours, KPI floors/ceilings (Settings page) |
| `analytics_daily` | Shift KPI snapshots (OEE components, uptime, flow) |
| `analytics_heatmaps` | 10×10 grid JSON per camera/shift/date |
| `health_metric_samples` | Telemetry history (backend latency probes) |

**DBeaver tip:** connect to the absolute path printed by `uv run python -c "from vision_ops_alerting.config import settings; print(settings.database_url)"`. Refresh schema (F5) after upgrades. While `run-local.sh` runs, SQLite uses WAL mode (`vision_ops.db-wal`).

### Startup seed behavior

On every startup (`init_db()`):

- Creates tables if missing (+ incremental SQLite migrations in `db/session.py`)
- Seeds **default admin user** if `users` is empty (`admin@visionops.local` / `admin123`)
- Seeds **default cameras** if `cameras` table is empty
- Ensures **built-in email templates** and **industrial reason codes**
- Ensures **default action rules** (one rule per case type)
- Ensures **default plant_config** row

If `ALERTING_SEED_DB=true` and `alert_rules` is empty, also inserts demo rules, events, and analytics rows (`db/seed.py`).

---

## API reference

Base URLs:

- Backend: `http://localhost:8000`
- Alerting: `http://localhost:8001`
- Via frontend proxy: `http://localhost:3000/vision-api/...` and `/alerting-api/...`

Interactive OpenAPI docs (when servers are running):

- http://localhost:8000/docs
- http://localhost:8001/docs

---

### vision-ops-backend (`:8000`)

#### Health

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

#### Cameras

```bash
# List webcam + industrial mocks (cam-01, cam-02 when VISION_ENABLED=true)
curl -s http://localhost:8000/api/cameras | jq .

# MJPEG stream (open in browser or <img src="...">)
open "http://localhost:8000/api/cameras/webcam-0/stream"
```

#### Face enrollment (SFace + YuNet)

```bash
curl -s http://localhost:8000/api/faces/status | jq .

curl -s http://localhost:8000/api/faces/storage | jq .

# Enroll (webcam must be running)
curl -X POST http://localhost:8000/api/faces/enroll \
  -H 'Content-Type: application/json' \
  -d '{"name":"Carlos"}'

# Remove enrollment
curl -X DELETE http://localhost:8000/api/faces/enroll
```

#### Vision probes (DINO + V-JEPA)

```bash
curl -s http://localhost:8000/api/vision/status | jq .

# Assembly camera — DINO heatmap
curl -X POST http://localhost:8000/api/vision/probe \
  -H 'Content-Type: application/json' \
  -d '{"camera_id":"cam-01","mode":"auto"}'

# Warehouse camera — V-JEPA anomaly
curl -X POST http://localhost:8000/api/vision/probe \
  -H 'Content-Type: application/json' \
  -d '{"camera_id":"cam-02","mode":"auto","set_baseline":true}'

# Fetch heatmap overlay JPEG
curl -o heatmap.jpg http://localhost:8000/api/vision/artifacts/cam-01/overlay
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/api/cameras` | Camera list with `streamUrl`, overlays |
| GET | `/api/cameras/{id}/stream` | MJPEG multipart (webcam-0 only) |
| GET | `/api/faces/status` | Enrollment state |
| GET | `/api/faces/storage` | Paths + privacy explanation |
| GET | `/api/faces/preview` | Enrollment preview JPEG |
| POST | `/api/faces/enroll` | Capture samples → embedding |
| DELETE | `/api/faces/enroll` | Remove enrollment |
| GET | `/api/vision/status` | Probe state for cam-01, cam-02 |
| GET | `/api/vision/storage` | Artifact paths |
| POST | `/api/vision/probe` | Run DINO or V-JEPA probe |
| GET | `/api/vision/artifacts/{id}/overlay` | Heatmap overlay image |
| GET | `/api/vision/artifacts/{id}/still` | Cached still frame |
| GET | `/api/vision/artifacts/{id}/preview` | Probe preview |
| GET | `/api/vision/cameras/{id}/last` | Last probe JSON |

---

### vision-ops-alerting (`:8001`)

#### Auth

```bash
curl -s -X POST http://localhost:8001/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@visionops.local","password":"admin123"}' | jq .

curl -s http://localhost:8001/api/auth/me \
  -H "Authorization: Bearer $TOKEN" | jq .
```

#### Health & telemetry

```bash
curl -s http://localhost:8001/health
# {"ok":true,"service":"vision-ops-alerting"}

curl -s http://localhost:8001/api/telemetry | jq .
curl -s http://localhost:8001/api/notifications/email/status | jq .
```

#### Cameras (SQLite + backend merge)

```bash
curl -s "http://localhost:8001/api/cameras" | jq .
curl -s "http://localhost:8001/api/cameras?status=live&zone=Assembly" | jq .
curl -s http://localhost:8001/api/cameras/stats/live | jq .

curl -X POST http://localhost:8001/api/cameras \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Dock Camera 3",
    "location": "Loading Dock",
    "zone": "Zone B",
    "sourceType": "rtsp",
    "streamUrl": "rtsp://example/stream",
    "inferenceModel": "yolov8",
    "backendCameraId": "cam-02"
  }'

curl -X DELETE http://localhost:8001/api/cameras/cam-xxxxxxxxxxxx
```

#### Alert rules

```bash
curl -s http://localhost:8001/api/alerts/actions | jq .
curl -s http://localhost:8001/api/alerts/rules | jq .

curl -X POST http://localhost:8001/api/alerts/rules \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "icon": "schedule",
    "title": "Idle operator Line 4",
    "description": "Alert when idle > 5 min",
    "zone": "LINE 4",
    "caseType": "user_not_working",
    "severity": "WARNING",
    "enabled": true,
    "notifyEmail": true
  }'

curl -X POST http://localhost:8001/api/alerts/rules/rule-xxxxxxxxxxxx/toggle
curl -X PATCH http://localhost:8001/api/alerts/rules/rule-xxxxxxxxxxxx \
  -H 'Content-Type: application/json' \
  -d '{"severity":"CRITICAL"}'
curl -X DELETE http://localhost:8001/api/alerts/rules/rule-xxxxxxxxxxxx

curl -s http://localhost:8001/api/alerts/deliveries | jq .
```

**Case types:** `user_not_working` · `user_left_position` · `forklift_in_zone` · `unknown`

#### Email dispatch (creates timeline event)

```bash
# Full pipeline: classify → rule check → email (or dry-run) → persist event
curl -X POST http://localhost:8001/api/alerting/email \
  -H 'Content-Type: application/json' \
  -d '{
    "site_id": "site-01",
    "line_id": "line-a",
    "camera_id": "cam-01",
    "timestamp": "2026-05-26T20:00:00Z",
    "actor": {"type": "operator", "track_id": "12", "name": "Operator 12"},
    "evidence": {"idle_seconds": 900}
  }'

# Test templates (uses built-in test contexts)
curl -X POST http://localhost:8001/api/alerting/email/test/user_not_working
curl -X POST http://localhost:8001/api/alerting/email/test/forklift_in_zone
```

**IndustrialContext schema** (`schemas.py`):

```json
{
  "site_id": "string",
  "line_id": "string",
  "camera_id": "string",
  "timestamp": "ISO-8601",
  "actor": { "type": "operator|forklift|unknown", "track_id": "?", "name": "?" },
  "evidence": { "idle_seconds": 900, "roi": "zone-id", "bbox": [0,0,1,1] },
  "links": { "live_url": "...", "timeline_url": "..." },
  "case_type": "optional override",
  "severity": "optional override"
}
```

#### Timeline (workflow)

```bash
curl -s "http://localhost:8001/api/timeline?limit=20" | jq .
curl -s "http://localhost:8001/api/timeline?severity=critical&resolutionStatus=OPEN" | jq .
curl -s http://localhost:8001/api/timeline/summary | jq .
curl -s http://localhost:8001/api/timeline/stats | jq .
curl -s http://localhost:8001/api/timeline/reason-codes | jq .

# Requires Authorization header (logged-in user name recorded on event)
curl -X PATCH http://localhost:8001/api/timeline/evt-xxxxxxxxxxxx/acknowledge \
  -H "Authorization: Bearer $TOKEN"

curl -X PATCH http://localhost:8001/api/timeline/evt-xxxxxxxxxxxx/resolve \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"status":"RESOLVED","reasonCode":"OPERATOR_IDLE","downtimeSeconds":120,"notes":"Retrained"}'
```

**Resolution statuses:** `OPEN` → `ACKNOWLEDGED` → `RESOLVED` | `FALSE_POSITIVE`

#### Analytics (industrial KPIs)

```bash
curl -s "http://localhost:8001/api/analytics/summary?shift=morning" | jq .
curl -s "http://localhost:8001/api/analytics/oee?shift=morning" | jq .
curl -s "http://localhost:8001/api/analytics/coq?shift=morning" | jq .
curl -s "http://localhost:8001/api/analytics/pareto?shift=morning" | jq .
curl -s "http://localhost:8001/api/analytics/heatmap?shift=morning&cameraId=cam-01" | jq .
curl -s "http://localhost:8001/api/analytics/insights?shift=morning" | jq .
```

KPI formulas read from `plant_config` (editable on `/settings`). UI shows ⓘ tooltips via `GET /api/settings/kpi-definitions`.

#### Plant settings

```bash
curl -s http://localhost:8001/api/settings/plant | jq .
curl -s http://localhost:8001/api/settings/kpi-definitions | jq .

curl -X PATCH http://localhost:8001/api/settings/plant \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"lineCostPerMinute":130,"siteName":"Plant A"}'
```

#### VisionOps AI Advisor

```bash
curl -s "http://localhost:8001/api/advisor/welcome?page=timeline" | jq .

curl -X POST http://localhost:8001/api/advisor/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Any open critical incidents?","page":"timeline","pageTitle":"Post-Shift Log"}' | jq .
```

#### Email templates

```bash
curl -s http://localhost:8001/api/alerts/email-templates | jq .
curl -X POST http://localhost:8001/api/alerts/email-templates/tmpl-xxx/preview | jq .
```

---

## Frontend map (`vision-ops-app`)

| Route | Component | Primary API (`lib/api.ts`) |
|-------|-----------|----------------------------|
| `/login` | `LoginPageClient` | `loginUser`, `registerUser` → alerting |
| `/live` | `LivePageClient` | `getLiveCameraFeeds`, `fetchLiveStats`, `fetchRealtimeEvents` |
| `/timeline` | `TimelinePageClient` | `fetchTimelineEvents`, ack/resolve, `fetchShiftSummary`, export |
| `/analytics` | `AnalyticsPageClient` | OEE, CoQ, Pareto, insights, heatmap, KPI tooltips |
| `/alerts` | `AlertsPageClient` | `fetchAlertRules`, CRUD, email templates, `sendTestAlertEmail` |
| `/settings` | `SettingsPageClient` | `fetchPlantSettings`, `updatePlantSettings`, KPI definitions |
| `/identity` | `IdentityEnrollmentPanel` | `fetchFaceStatus`, `enrollFace` → backend via `/vision-api` |
| `/vision-lab` | `VisionLabPanel` | `fetchVisionStatus`, `runVisionProbe` → backend |
| *(all dashboard)* | `VisionOpsAdvisor` | `fetchAdvisorWelcome`, `advisorChat` → alerting |
| *(TopNav)* | `NotificationsBell` | Open `OPEN` events count + link to timeline |

**Auth:** `(dashboard)/layout.tsx` wraps pages in `AuthProvider`; JWT in `localStorage` via `lib/auth.ts`.

**Important files for agents:**

| File | Why |
|------|-----|
| `vision-ops-app/lib/api.ts` | All HTTP client functions + proxy + JWT headers |
| `vision-ops-app/lib/auth.ts` | Token storage |
| `vision-ops-app/components/advisor/VisionOpsAdvisor.tsx` | AI chat bot UI |
| `vision-ops-app/next.config.ts` | API rewrites |
| `vision-ops-app/AGENTS.md` | Next.js 16 breaking changes — read before editing UI |
| `vision-ops-alerting/src/vision_ops_alerting/db/models.py` | DB schema |
| `vision-ops-alerting/src/vision_ops_alerting/agent.py` | Alert classification + email |
| `vision-ops-alerting/src/vision_ops_alerting/advisor_agent.py` | Floor advisor chat |
| `vision-ops-alerting/src/vision_ops_alerting/services/event_workflow.py` | Timeline ack/resolve |
| `vision-ops-alerting/src/vision_ops_alerting/services/industrial_analytics.py` | OEE / CoQ / Pareto |
| `vision-ops-backend/src/vision_ops_backend/main.py` | Backend lifespan (webcam) |

---

## Gitignored / local-only assets

| Path | Contents |
|------|----------|
| `models/face_detection_yunet/*.onnx` | YuNet face detector (~1 MB) |
| `models/face_recognition_sface/*.onnx` | SFace recognizer (~40 MB total download) |
| `vision-ops-backend/data/faces/` | `owner.npz`, enrollment preview |
| `vision-ops-backend/data/vision/` | DINO heatmaps, V-JEPA probe JSON |
| `vision-ops-alerting/data/` | SQLite `vision_ops.db` (+ WAL sidecars) |
| `vision-ops-app/.env.local` | Local env |
| `vision-ops-alerting/.env` | MailerSend token, auth secret, recipients |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **401 on ack/resolve/settings** | Sign in at `/login`; mutating APIs require `Authorization: Bearer` |
| **Advisor says unavailable** | Start Ollama + `llama3.1`; alerting must be on :8001 |
| **503 on webcam stream** | Grant camera permission to Terminal/IDE; close other apps using the camera |
| **Black Live tile** | Start backend before refreshing; check `GET /api/cameras` |
| **CORS errors** | Add your UI origin to `CORS_ORIGINS` (backend) and `ALERTING_CORS_ORIGINS` |
| **Email not sending** | Set `ALERTING_DRY_RUN=false` + valid `ALERTING_MAILERSEND_API_TOKEN`; check `/api/notifications/email/status` |
| **403 on email dispatch** | Enable a matching `alert_rule` for the classified `case_type` |
| **Empty timeline** | Send test email or set `ALERTING_SEED_DB=true` and restart |
| **`users` table missing in DBeaver** | Refresh connection (F5); use absolute DB path from `ALERTING_DATABASE_URL` |
| **Classification always fallback** | Start Ollama with `llama3.1` or pass explicit `case_type` in payload |
| **Port in use** | `./run-local.sh` kills processes on 8000/8001/3000; or set `BACKEND_PORT`, `ALERTING_PORT`, `FRONTEND_PORT` |

---

## For AI coding agents

When starting a new session on this repo:

1. **Read this README** for architecture, ports, and API contracts.
2. **Run `./run-local.sh`** to validate the full stack (requires uv + npm + camera on Mac).
3. **Pick the right service:**
   - UI changes → `vision-ops-app/` (read `AGENTS.md` first — Next.js 16 differs from training data)
   - Webcam / vision / face → `vision-ops-backend/`
   - Rules / timeline / email / DB / auth / advisor → `vision-ops-alerting/`
4. **Auth:** login API + `users` table in alerting; frontend JWT in `lib/auth.ts`. No RBAC yet.
5. **Camera changes:** Update alerting `cameras` table **and** backend runtime if streams/probes are involved; Live page reads merged data from alerting.
6. **New alert case types:** Add to `schemas.py` CaseType, `templates.py`, `alert_actions.py`, seed rules, and frontend types in `lib/api.ts`.
7. **KPI changes:** Edit `plant_config` + `services/industrial_analytics.py`; expose defs in `services/plant_settings.py` KPI_DEFINITIONS.
8. **Do not commit** `.env`, `.env.local`, SQLite DB, face embeddings, or API tokens.

Sub-project READMEs (shorter): `vision-ops-backend/README.md`, `vision-ops-alerting/README.md`.

---

## Research stack (root `pyproject.toml`)

The repo root Python environment supports notebooks and upstream model code under `models/`:

| Component | Location | Role |
|-----------|----------|------|
| **DINOv3** | `models/dinov3-main/` | Spatial SSL representations |
| **V-JEPA 2.x** | `models/vjepa2-main/` | Temporal latent prediction |
| **InHARD** | `data_sample/InHARD-master/` | Industrial action dataset reference |
| **YOLOv8 / DeepSORT** | root deps | Prototype tracking baselines |
| **Strands + Ollama** | `vision-ops-alerting/` | Alert case classifier + VisionOps AI Advisor chat |
| **MailerSend** | `vision-ops-alerting/` | Transactional alert email |

Model weights (`.pt`, `.onnx`, `.pth`) are excluded from git. Vision probes in the demo backend optionally use torch/transformers when installed.

```bash
uv sync --all-groups          # full research deps
uv sync --group notebooks     # Jupyter only
```

---

## Licencia

Privado — © Alignity IQ Edge, LLC. Todos los derechos reservados. El código de terceros bajo `models/` conserva sus licencias originales.
