from __future__ import annotations

import asyncio
import html
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from arq.connections import ArqRedis, create_pool
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from explicit_content_cut import ExplicitCutPlan

from .config import load_settings, to_redis_settings
from .explicit_cut_service import apply_plan, create_plan
from .models import (
    ExplicitCutPlanCreatedResponse,
    ExplicitCutPlanDecisionRequest,
    ExplicitCutPlanRecord,
    ExplicitCutPlanRequest,
    JobCreatedResponse,
    JobRecord,
    TranscribeJobRequest,
)
from .plan_store import SqliteExplicitCutPlanStore
from .store import RedisJobStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

settings = load_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = None
    plan_store = SqliteExplicitCutPlanStore(settings.explicit_cut_plan_db_path)
    plan_store.initialize()
    app.state.explicit_cut_plan_store = plan_store
    try:
        app.state.redis = await create_pool(to_redis_settings(settings))
    except Exception as error:
        logger.warning('Redis unavailable during app startup: %s', error)
        app.state.redis = None
    try:
        yield
    finally:
        redis: ArqRedis | None = app.state.redis
        if redis is not None:
            await redis.aclose()


app = FastAPI(title='Znayka Integration Service', version='0.2.0', lifespan=lifespan)


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/', response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    jobs = await _list_jobs_for_home(request)
    plans = await asyncio.to_thread(_plan_store(request).list_plans, 20)
    return HTMLResponse(_render_home(jobs, plans))


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


@app.post('/explicit-cut/plans/form')
async def create_explicit_cut_plan_from_form(
    request: Request,
    video_name: str = Form(...),
    frame_verification_backend: str = Form(default='off'),
    force_replan: bool = Form(default=False),
) -> RedirectResponse:
    payload = ExplicitCutPlanRequest(
        video_name=video_name,
        frame_verification_backend=frame_verification_backend or None,
        force_replan=force_replan,
    )
    await _create_explicit_cut_plan_record(request, payload)
    return RedirectResponse(url='/', status_code=303)


@app.get('/jobs', response_model=list[JobRecord])
async def list_jobs(request: Request) -> list[JobRecord]:
    return await _store(request).list_jobs()


@app.get('/jobs/{job_id}', response_model=JobRecord)
async def get_job(job_id: str, request: Request) -> JobRecord:
    job = await _store(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return job


@app.post(
    '/explicit-cut/plans',
    response_model=ExplicitCutPlanCreatedResponse,
    status_code=201,
)
async def create_explicit_cut_plan(
    payload: ExplicitCutPlanRequest,
    request: Request,
) -> ExplicitCutPlanCreatedResponse:
    record = await _create_explicit_cut_plan_record(request, payload)
    return ExplicitCutPlanCreatedResponse(id=record.id, status=record.status)


async def _create_explicit_cut_plan_record(
    request: Request,
    payload: ExplicitCutPlanRequest,
) -> ExplicitCutPlanRecord:
    store = _plan_store(request)
    plan = await asyncio.to_thread(create_plan, settings, payload)
    now = _now_utc()
    record = ExplicitCutPlanRecord(
        id=uuid4().hex,
        status='planned',
        created_at=now,
        updated_at=now,
        request=payload,
        plan=plan.to_dict(),
        result_message='Explicit-cut plan created',
    )
    await asyncio.to_thread(store.add_plan, record)
    return record


@app.get('/explicit-cut/plans', response_model=list[ExplicitCutPlanRecord])
async def list_explicit_cut_plans(request: Request) -> list[ExplicitCutPlanRecord]:
    return await asyncio.to_thread(_plan_store(request).list_plans, 100)


@app.get('/explicit-cut/plans/{plan_id}', response_model=ExplicitCutPlanRecord)
async def get_explicit_cut_plan(plan_id: str, request: Request) -> ExplicitCutPlanRecord:
    record = await asyncio.to_thread(_plan_store(request).get_plan, plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Explicit-cut plan not found')
    return record


@app.post('/explicit-cut/plans/{plan_id}/approve', response_model=ExplicitCutPlanRecord)
async def approve_explicit_cut_plan(
    plan_id: str,
    payload: ExplicitCutPlanDecisionRequest,
    request: Request,
) -> ExplicitCutPlanRecord:
    store = _plan_store(request)
    current = await asyncio.to_thread(store.get_plan, plan_id)
    if current is None:
        raise HTTPException(status_code=404, detail='Explicit-cut plan not found')
    next_status = 'approved' if payload.approved else 'rejected'
    return await asyncio.to_thread(
        store.update_plan,
        plan_id,
        status=next_status,
        note=payload.note,
        result_message=(
            'Explicit-cut plan approved' if payload.approved else 'Explicit-cut plan rejected'
        ),
    )


@app.post('/explicit-cut/plans/{plan_id}/decision')
async def decide_explicit_cut_plan_from_form(
    plan_id: str,
    request: Request,
    approved: bool = Form(...),
    note: str = Form(default=''),
) -> RedirectResponse:
    await approve_explicit_cut_plan(
        plan_id=plan_id,
        payload=ExplicitCutPlanDecisionRequest(approved=approved, note=note or None),
        request=request,
    )
    return RedirectResponse(url='/', status_code=303)


@app.post('/explicit-cut/plans/{plan_id}/apply', response_model=ExplicitCutPlanRecord)
async def apply_explicit_cut_plan_endpoint(
    plan_id: str,
    request: Request,
) -> ExplicitCutPlanRecord:
    store = _plan_store(request)
    current = await asyncio.to_thread(store.get_plan, plan_id)
    if current is None:
        raise HTTPException(status_code=404, detail='Explicit-cut plan not found')
    if current.status != 'approved':
        raise HTTPException(
            status_code=409,
            detail='Explicit-cut plan must be approved before apply.',
        )

    applied_plan = await asyncio.to_thread(
        apply_plan,
        settings,
        current.request,
        ExplicitCutPlan.from_dict(current.plan),
    )
    return await asyncio.to_thread(
        store.update_plan,
        plan_id,
        status='applied',
        plan=applied_plan.to_dict(),
        result_message='Explicit-cut plan applied',
    )


@app.post('/explicit-cut/plans/{plan_id}/apply/form')
async def apply_explicit_cut_plan_from_form(
    plan_id: str,
    request: Request,
) -> RedirectResponse:
    await apply_explicit_cut_plan_endpoint(plan_id=plan_id, request=request)
    return RedirectResponse(url='/', status_code=303)


def _store(request: Request) -> RedisJobStore:
    redis: ArqRedis | None = request.app.state.redis
    if redis is None:
        raise HTTPException(status_code=503, detail='Redis is unavailable')
    return RedisJobStore(redis)


def _plan_store(request: Request) -> SqliteExplicitCutPlanStore:
    return request.app.state.explicit_cut_plan_store


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


async def _list_jobs_for_home(request: Request) -> list[JobRecord]:
    redis: ArqRedis | None = request.app.state.redis
    if redis is None:
        return []
    return await RedisJobStore(redis).list_jobs(limit=20)


def _render_home(jobs: list[JobRecord], plans: list[ExplicitCutPlanRecord]) -> str:
    job_rows = ''.join(
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
    if not job_rows:
        job_rows = '<tr><td colspan="4">No jobs yet</td></tr>'

    plan_rows = ''.join(_render_plan_row(plan) for plan in plans)
    if not plan_rows:
        plan_rows = '<tr><td colspan="7">No explicit-cut plans yet</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Znayka Worker Control</title>
  <style>
    body {{ font-family: "Segoe UI", Tahoma, sans-serif; margin: 24px; }}
    fieldset {{ margin-bottom: 12px; max-width: 900px; }}
    label {{ display: block; margin: 6px 0; }}
    input[type="text"] {{ width: 420px; max-width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .inline {{ display: inline-block; margin-right: 20px; }}
    .btn {{ padding: 8px 14px; font-size: 14px; }}
    .btn-small {{ padding: 6px 10px; font-size: 12px; }}
    .stack {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .muted {{ color: #666; }}
    .mono {{ font-family: Consolas, monospace; }}
    textarea {{ width: 320px; max-width: 100%; min-height: 52px; }}
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

  <h2>Explicit Cut Plans</h2>
  <form method="post" action="/explicit-cut/plans/form">
    <fieldset>
      <legend>Create Explicit-Cut Plan</legend>
      <label>Video name: <input type="text" name="video_name" required /></label>
      <label>Frame verification backend:
        <select name="frame_verification_backend">
          <option value="off" selected>off</option>
          <option value="opennsfw2">opennsfw2</option>
          <option value="nudenet">nudenet</option>
        </select>
      </label>
      <label><input type="checkbox" name="force_replan" /> Force re-plan</label>
      <button class="btn" type="submit">Create Plan</button>
      <div class="muted">Low-JS operator flow: create, review, approve or reject, then apply.</div>
    </fieldset>
  </form>

  <table>
    <thead><tr><th>Plan ID</th><th>Status</th><th>Video</th><th>Cut</th><th>Verification</th><th>Updated</th><th>Actions</th></tr></thead>
    <tbody>{plan_rows}</tbody>
  </table>

  <h2>Recent Jobs</h2>
  <table>
    <thead><tr><th>Job ID</th><th>Status</th><th>Updated</th><th>Video</th></tr></thead>
    <tbody>{job_rows}</tbody>
  </table>
</body>
</html>"""


def _render_plan_row(plan: ExplicitCutPlanRecord) -> str:
    cut_spans = plan.plan.get('cut_spans', [])
    cut_summary = (
        f"{plan.plan.get('cut_duration_sec', 0)}s across {len(cut_spans)} cut(s)"
        if cut_spans
        else 'No cuts proposed'
    )
    verification = plan.plan.get('frame_verification_summary') or 'subtitle-only'
    note_value = html.escape(plan.note or '')
    video_name = html.escape(plan.request.video_name)
    action_forms = ''.join(
        [
            (
                f'<form method="post" action="/explicit-cut/plans/{html.escape(plan.id)}/decision">'
                '<input type="hidden" name="approved" value="true" />'
                f'<textarea name="note" placeholder="Approval note">{note_value}</textarea>'
                '<button class="btn-small" type="submit">Approve</button>'
                '</form>'
            )
            if plan.status == 'planned'
            else ''
        ]
        + [
            (
                f'<form method="post" action="/explicit-cut/plans/{html.escape(plan.id)}/decision">'
                '<input type="hidden" name="approved" value="false" />'
                f'<textarea name="note" placeholder="Rejection note">{note_value}</textarea>'
                '<button class="btn-small" type="submit">Reject</button>'
                '</form>'
            )
            if plan.status == 'planned'
            else ''
        ]
        + [
            (
                f'<form method="post" action="/explicit-cut/plans/{html.escape(plan.id)}/apply/form">'
                '<button class="btn-small" type="submit">Apply</button>'
                '</form>'
            )
            if plan.status == 'approved'
            else ''
        ]
    )
    if not action_forms:
        action_forms = '<span class="muted">No actions</span>'
    return (
        '<tr>'
        f'<td class="mono">{html.escape(plan.id)}</td>'
        f'<td>{html.escape(plan.status)}</td>'
        f'<td>{video_name}</td>'
        f'<td>{html.escape(cut_summary)}</td>'
        f'<td>{html.escape(str(verification))}</td>'
        f'<td>{html.escape(plan.updated_at.isoformat())}</td>'
        f'<td><div class="stack">{action_forms}</div></td>'
        '</tr>'
    )
