# vision-ops-alerting

Alerting backend for VisionOps: classifies industrial events, sends email via MailerSend, and persists rules/events/analytics in a local **SQLite** database.

## Local run

```bash
cd vision-ops-alerting
cp .env.example .env   # set MAILERSEND token + recipients
uv sync
uv run uvicorn vision_ops_alerting.main:app --reload --host 0.0.0.0 --port 8001
```

Or from repo root: `./run-local.sh` (starts backend **8000**, alerting **8001**, frontend **3000**).

## Database schema (SQLite)

File: `data/vision_ops.db` (gitignored, auto-created on startup)

| Table | Purpose |
|-------|---------|
| `alert_rules` | Configurable vision alert rules (`/alerts` page) |
| `events` | Timeline incidents from cameras + email triggers |
| `alert_deliveries` | Email send log linked to events |
| `analytics_daily` | Shift-level KPI snapshots |
| `analytics_heatmaps` | Heatmap grid data per camera/shift |
| `har_watch_sessions` | Live HAR watch periods per camera/video loop |
| `har_activity_logs` | Integral per-camera action log (live + probe) |
| `har_inference_runs` / `har_inference_results` | Batch probe history |

Seed data (mock rules + timeline events) is inserted on first run when `ALERTING_SEED_DB=true`.

### HAR activity ingest (from vision-ops-backend live loop)

```bash
curl -X POST http://localhost:8001/api/har/activity \
  -H 'Content-Type: application/json' \
  -d '{"entry":{"camera_id":"cam-har-01","model_id":"dinov2-puro","source":"live","prediction":{"label":"Assemble system","confidence":0.82}}}'
```

Query logs: `GET /api/har/activity?cameraId=cam-har-01`  
Analytics: `GET /api/har/analytics/daily?cameraId=cam-har-01`  
Per-camera chat: `POST /api/advisor/camera-chat` with `{ "cameraId", "message" }`.

Non-assembly HAR actions promote to Timeline as `har_action_deviation` events. Email remains dry-run (`ALERTING_HAR_EMAIL_ENABLED=false` by default).

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| POST | `/api/alerting/email` | Classify + send email + persist event + delivery log |
| GET | `/api/alerts/rules` | List alert rules |
| POST | `/api/alerts/rules` | Create rule |
| PATCH | `/api/alerts/rules/{id}` | Update rule |
| DELETE | `/api/alerts/rules/{id}` | Delete rule |
| POST | `/api/alerts/rules/{id}/toggle` | Enable/disable rule |
| GET | `/api/alerts/deliveries` | Recent email send logs |
| GET | `/api/timeline` | Timeline events |
| GET | `/api/timeline/summary` | Shift summary sidebar |
| GET | `/api/timeline/{id}` | Single event |
| GET | `/api/analytics/summary` | Flow efficiency / uptime |
| GET | `/api/analytics/heatmap` | Heatmap grid |
| GET | `/api/analytics/insights` | Downtime + bottlenecks |

## Frontend integration

Next.js proxies alerting via `/alerting-api/*` → `http://localhost:8001`.

Pages wired to API (when `NEXT_PUBLIC_USE_MOCK_DATA` is not `true`):

- `/alerts` — rules CRUD + toggle
- `/timeline` — events + shift summary
- `/analytics` — summary, heatmap, insights

## Test email + DB persistence

```bash
curl -X POST http://localhost:8001/api/alerting/email \
  -H 'Content-Type: application/json' \
  -d '{
    "site_id":"site-01","line_id":"line-a","camera_id":"cam-01",
    "timestamp":"2026-05-26T20:00:00Z",
    "actor":{"type":"operator","track_id":"12","name":"Operator 12"},
    "evidence":{"idle_seconds":900}
  }'
```

Response includes `event_id` and `delivery_id`; the event appears on `/timeline`.
