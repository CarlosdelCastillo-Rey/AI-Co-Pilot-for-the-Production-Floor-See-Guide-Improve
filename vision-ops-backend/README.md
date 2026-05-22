# vision-ops-backend

Dev backend for **VisionOps Live**: streams your Mac webcam as MJPEG so the first tile on `/live` behaves like a connected IP camera (RTSP comes later).

## Quick start

**Both backend + frontend (recommended):** from the repo root:

```bash
./run-local.sh
```

**Backend only:**

```bash
cd vision-ops-backend
uv sync
uv run uvicorn vision_ops_backend.main:app --reload --host 0.0.0.0 --port 8000
```

Optional: `cp .env.example .env` and set `CAMERA_INDEX`, `PUBLIC_API_BASE`.

## Face models (Phase 2)

From repo root (once):

```bash
./models/install_face_models.sh
```

Uses [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet) + [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface).

Enroll your face (webcam must be running):

```bash
curl -X POST http://localhost:8000/api/faces/enroll
```

Or open **My Identity** in the app: http://localhost:3000/identity

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/api/cameras` | Camera metadata + `streamUrl` + overlays |
| GET | `/api/cameras/webcam-0/stream` | MJPEG multipart stream (boxes drawn on stream) |
| GET | `/api/faces/status` | Enrollment state |
| POST | `/api/faces/enroll` | Capture webcam samples → owner embedding |

## Frontend (port 3000)

In `vision-ops-app`:

```bash
cp .env.local.example .env.local
npm run dev
```

Next.js runs at **http://localhost:3000** (Network URL e.g. `http://192.168.0.113:3000` also works if CORS/API base are updated — see `.env.example` comments).

Open **http://localhost:3000/live** — **Camera 01** shows the webcam; other cameras stay static mocks.

| Service | Port | URL |
|---------|------|-----|
| vision-ops-app | **3000** | http://localhost:3000 |
| vision-ops-backend | **8000** | http://localhost:8000 |

## Troubleshooting

- **503 on stream:** another app is using the camera, or macOS denied camera access to Terminal/IDE.
- **Black tile:** start the backend before refreshing the page.
- **CORS errors:** ensure `CORS_ORIGINS` includes your Next.js origin.

## Later (main VisionOps plan)

RTSP ingest, YOLO overlays, events API — see repo Fase 1 plan.
