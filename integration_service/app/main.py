from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from uuid import uuid4

from arq.connections import ArqRedis, create_pool
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .config import load_settings, to_redis_settings
from .models import JobCreatedResponse, JobRecord, TranscribeJobRequest
from .store import RedisJobStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

settings = load_settings()
app = FastAPI(title='Znayka Integration Service', version='0.2.0')


@app.on_event('startup')
async def startup_event() -> None:
    app.state.redis = await create_pool(to_redis_settings(settings))


@app.on_event('shutdown')
async def shutdown_event() -> None:
    redis: ArqRedis = app.state.redis
    await redis.aclose()


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/', response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    store = _store(request)
    jobs = await store.list_jobs(limit=20)
    return HTMLResponse(_render_home(jobs))


@app.post('/jobs/form')
async def create_job_from_form(
    request: Request,
    video_name: str = Form(default=''),
    whisper_language: str = Form(default=''),
    translation_source_language: str = Form(default=''),
    translation_target_language: str = Form(default=''),
    offline_mode: bool = Form(default=False),
    dry_run: bool = Form(default=False),
    run_generate_spans: bool = Form(default=False),
    run_filter_sidecars: bool = Form(default=False),
    run_transcription: bool = Form(default=False),
    run_translation: bool = Form(default=False),
    run_merge: bool = Form(default=False),
    run_sidecar_replace: bool = Form(default=False),
    run_bake_subtitles: bool = Form(default=False),
) -> RedirectResponse:
    payload = TranscribeJobRequest(
        video_name=video_name or None,
        whisper_language=whisper_language or None,
        translation_source_language=translation_source_language or None,
        translation_target_language=translation_target_language or None,
        offline_mode=offline_mode,
        dry_run=dry_run,
        run_generate_spans=run_generate_spans,
        run_filter_sidecars=run_filter_sidecars,
        run_transcription=run_transcription,
        run_translation=run_translation,
        run_merge=run_merge,
        run_sidecar_replace=run_sidecar_replace,
        run_bake_subtitles=run_bake_subtitles,
    )
    await _enqueue_job(request, payload)
    return RedirectResponse(url='/', status_code=303)


@app.post('/jobs/transcribe', response_model=JobCreatedResponse, status_code=202)
async def create_transcribe_job(
    payload: TranscribeJobRequest,
    request: Request,
) -> JobCreatedResponse:
    job = await _enqueue_job(request, payload)
    return JobCreatedResponse(id=job.id, status=job.status)


@app.get('/jobs', response_model=list[JobRecord])
async def list_jobs(request: Request) -> list[JobRecord]:
    return await _store(request).list_jobs()


@app.get('/jobs/{job_id}', response_model=JobRecord)
async def get_job(job_id: str, request: Request) -> JobRecord:
    job = await _store(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return job


def _store(request: Request) -> RedisJobStore:
    redis: ArqRedis = request.app.state.redis
    return RedisJobStore(redis)


async def _enqueue_job(request: Request, payload: TranscribeJobRequest) -> JobRecord:
    job_id = uuid4().hex
    now = _now_utc()
    job = JobRecord(
        id=job_id,
        status='queued',
        created_at=now,
        updated_at=now,
        request=payload,
    )

    redis: ArqRedis = request.app.state.redis
    store = RedisJobStore(redis)
    await store.add_job(job)

    queued = await redis.enqueue_job(
        'run_pipeline_job_task',
        job_id,
        payload.model_dump(mode='json'),
        _job_id=job_id,
    )
    if queued is None:
        raise HTTPException(status_code=409, detail='Job with this id already exists')
    return job


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _render_home(jobs: list[JobRecord]) -> str:
    rows = ''.join(
        (
            '<tr>'
            f'<td>{html.escape(job.id)}</td>'
            f'<td>{html.escape(job.status)}</td>'
            f'<td>{html.escape(job.updated_at.isoformat())}</td>'
            f'<td>{html.escape(job.request.video_name or "-")}</td>'
            '</tr>'
        )
        for job in jobs
    )
    if not rows:
        rows = '<tr><td colspan="4">No jobs yet</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Znayka Worker Control</title>
  <style>
    body {{ font-family: "Segoe UI", Tahoma, sans-serif; margin: 24px; }}
    fieldset {{ margin-bottom: 12px; max-width: 760px; }}
    label {{ display: block; margin: 6px 0; }}
    input[type="text"] {{ width: 420px; max-width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .inline {{ display: inline-block; margin-right: 20px; }}
    .btn {{ padding: 8px 14px; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>Znayka Worker Control</h1>
  <form method="post" action="/jobs/form">
    <fieldset>
      <legend>Video and Language</legend>
      <label>Video name (optional): <input type="text" name="video_name" /></label>
      <label>Whisper language: <input type="text" name="whisper_language" value="fr" /></label>
      <label>Translation source language: <input type="text" name="translation_source_language" value="fr" /></label>
      <label>Translation target language: <input type="text" name="translation_target_language" value="en" /></label>
    </fieldset>

    <fieldset>
      <legend>Job Options</legend>
      <label class="inline"><input type="checkbox" name="offline_mode" /> Offline mode</label>
      <label class="inline"><input type="checkbox" name="dry_run" checked /> Dry run</label>
    </fieldset>

    <fieldset>
      <legend>Pipeline Steps</legend>
      <label><input type="checkbox" name="run_generate_spans" /> Generate speech spans</label>
      <label><input type="checkbox" name="run_filter_sidecars" /> Filter sidecars</label>
      <label><input type="checkbox" name="run_transcription" checked /> Run transcription</label>
      <label><input type="checkbox" name="run_translation" checked /> Run translation</label>
      <label><input type="checkbox" name="run_merge" /> Merge subtitles</label>
      <label><input type="checkbox" name="run_sidecar_replace" /> Replace sidecar</label>
      <label><input type="checkbox" name="run_bake_subtitles" /> Bake subtitles</label>
    </fieldset>

    <button class="btn" type="submit">Queue Job</button>
  </form>

  <h2>Recent Jobs</h2>
  <table>
    <thead><tr><th>Job ID</th><th>Status</th><th>Updated</th><th>Video</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
