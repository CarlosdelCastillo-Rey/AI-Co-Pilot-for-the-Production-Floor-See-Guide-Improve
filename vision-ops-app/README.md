# vision-ops-app

Next.js 16 supervisor dashboard for **VisionOps** — connects to the unified backend on port **8000**.

## Quick start

From repo root:

```bash
./run-local.sh
```

Or frontend only:

```bash
npm install
cp .env.local.example .env.local   # if present; else set NEXT_PUBLIC_API_URL
npm run dev
```

Default: http://localhost:3000 — login `admin@visionops.local` / `admin123`

## Routes (current)

| Route | Purpose |
|-------|---------|
| `/login` | Email + password auth |
| `/analytics` | **Home** — OEE, CoQ, Pareto, heatmap, HAR model performance |
| `/live` | Mock camera wall, live HAR, bench controls, batch model probes, camera chat |
| `/har-hitl` | Person HITL — sessions, registry, review queue, tuning |
| `/timeline` | Post-shift log — ack / resolve workflow |
| `/alerts` | Alert rules + email templates |
| `/settings` | Plant KPI variables and formulas |

Redirects: `/har-analysis` → `/analytics`, `/vision-lab` → `/live`.

## API client

| File | Role |
|------|------|
| `lib/api.ts` | Unified REST client (`getApiFetchBase`, `fetchApi`, JWT via `authHeaders`) |
| `lib/har-v2-api.ts` | HAR v2 HITL endpoints (`/api/har/v2/...`) |
| `lib/auth.ts` | JWT in `localStorage` |
| `next.config.ts` | Rewrites `/api/*` → `NEXT_PUBLIC_API_URL` |

Browser calls use relative `/api/...` paths (Next.js proxy). SSR uses `NEXT_PUBLIC_API_URL` directly.

## Key components

| Area | Path |
|------|------|
| Live wall | `components/live/LivePageClient.tsx` |
| HITL | `components/har-v2/HarPersonHitl*.tsx` |
| Dashboard | `components/analytics/AnalyticsPageClient.tsx` |
| Advisor bot | `components/advisor/VisionOpsAdvisor.tsx` |
| Shell / nav | `components/layout/AppShell.tsx`, `Sidebar.tsx` |

Read `AGENTS.md` before editing — Next.js 16 conventions differ from older versions.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for SSR |
