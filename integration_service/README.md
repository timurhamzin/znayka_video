# Integration Service (Separate Project)

This service wraps the existing `znayka-video` script pipeline with a small
job API. It is intentionally separate so it can be moved into a future
monorepo backend later.

## What It Does

- Accepts transcription job requests over HTTP.
- Runs the existing root `main.py` pipeline in background.
- Persists job status in `integration_service/data/jobs.json`.

## Run

From repository root:

```powershell
cd integration_service
uv sync
uv run uvicorn app.main:app --reload --port 8010
```

Health check:

```bash
curl http://localhost:8010/health
```

## API

- `GET /health`
- `POST /jobs/transcribe`
- `GET /jobs`
- `GET /jobs/{job_id}`

Example payload:

```json
{
  "video_name": "Episode 01.mp4",
  "whisper_language": "fr",
  "translation_source_language": "fr",
  "translation_target_language": "en",
  "offline_mode": false,
  "dry_run": true
}
```

## Notes

- The service sets pipeline step flags to run transcription + translation.
- Existing root `.env` is still used as baseline.
- Job persistence is JSON-only MVP. Production storage should be PostgreSQL.
