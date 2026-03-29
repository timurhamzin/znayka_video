# Integration Service (App + Worker + Redis)

This is a separate project that wraps the root `znayka-video` pipeline into a
queue-based service.

Architecture:
- `app` (FastAPI): UI + API, enqueues jobs.
- `worker` (ARQ): runs pipeline jobs in a separate process.
- `redis`: queue and job state store.
- `sqlite`: durable explicit-cut approval plan store for the service layer.

## Why ARQ

ARQ is an asyncio-native task manager. It supports async tasks now and keeps the
door open for future async I/O-heavy integrations.

## Quick Setup (Local, PyCharm-friendly)

1. Copy env file:

```powershell
cd integration_service
Copy-Item .env.example .env
```

2. Install dependencies:

```powershell
uv sync
```

3. Start Redis:

```powershell
docker run --name znayka-redis -p 6379:6379 -d redis:7-alpine
```

4. Run app from PyCharm:
- Run script: `integration_service/scripts/local/run_app.py`
- Working directory: repo root

5. Run worker as a second process:
- Run script: `integration_service/scripts/local/run_worker.py`
- Working directory: repo root

6. Open UI:
- [http://127.0.0.1:8010](http://127.0.0.1:8010)

## API and UI

- `GET /` HTML form with tickboxes for pipeline steps.
- `GET /` low-JS operator page for jobs and explicit-cut plan review.
- `GET /health`
- `POST /jobs/transcribe`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /explicit-cut/plans`
- `GET /explicit-cut/plans`
- `GET /explicit-cut/plans/{plan_id}`
- `POST /explicit-cut/plans/{plan_id}/approve`
- `POST /explicit-cut/plans/{plan_id}/apply`

Headless explicit-cut flow:
1. `POST /explicit-cut/plans` to create a proposed cut plan.
2. Review the returned plan and evidence.
3. `POST /explicit-cut/plans/{plan_id}/approve` with `approved=true`.
4. `POST /explicit-cut/plans/{plan_id}/apply` only after approval.

Example:

```bash
curl -X POST http://127.0.0.1:8010/explicit-cut/plans \
  -H "Content-Type: application/json" \
  -d '{"video_name":"Toy.Soldiers.1991.720p.BluRay.x264-[YTS.AG].mp4"}'
```

Operator UI flow:
1. Open `/`
2. Create an explicit-cut plan with the simple HTML form
3. Review the summary row
4. Approve or reject
5. Apply only after approval

The form includes:
- language fields
- `dry_run` and `offline_mode`
- step tickboxes (`generate_spans`, `transcription`, `translation`, etc.)

## Docker Compose (App + Worker + Redis)

From `integration_service/`:

```bash
cp .env.example .env
docker compose up --build
```

Services:
- `redis` on `6379`
- `app` on `8010`
- `worker` consuming queued jobs

## Notes

- Worker executes the root pipeline entrypoint (`main.py`) via:
  - `uv run --project <repo_root> python main.py`
- Existing root `.env` remains the baseline for pipeline behavior.
- Explicit-cut plans are not stored in Redis. The service persists them in
  `INTEGRATION_EXPLICIT_CUT_PLAN_DB` so approval state survives worker/app
  restarts independently from the queue.
- The explicit-cut API uses shared Python logic directly (`plan` then `apply`)
  instead of report files as its source of truth.
- The explicit-cut core stays synchronous and reusable for CLI/library use.
  The service calls that work asynchronously from the HTTP layer so long-running
  planning/apply work does not block the event loop.
- CPU-friendly default is still subtitle-first planning with
  `frame_verification_backend=off`. If frame verification is enabled, keep it
  scoped to candidate spans only.
- This project stays separate for now and can be moved into monorepo backend
  later.
