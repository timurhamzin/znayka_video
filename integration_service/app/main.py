from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException

from .config import load_settings
from .models import JobCreatedResponse, JobRecord, TranscribeJobRequest
from .runner import run_pipeline_job
from .store import JobStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

settings = load_settings()
store = JobStore(settings.jobs_file)

app = FastAPI(title='Znayka Integration Service', version='0.1.0')


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/jobs/transcribe', response_model=JobCreatedResponse, status_code=202)
def create_transcribe_job(
    payload: TranscribeJobRequest,
    background_tasks: BackgroundTasks,
) -> JobCreatedResponse:
    job_id = uuid4().hex
    now = _now_utc()
    job = JobRecord(
        id=job_id,
        status='queued',
        created_at=now,
        updated_at=now,
        request=payload,
    )
    store.add_job(job)
    background_tasks.add_task(_execute_job, job_id, payload)
    return JobCreatedResponse(id=job_id, status='queued')


@app.get('/jobs', response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return store.list_jobs()


@app.get('/jobs/{job_id}', response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return job


def _execute_job(job_id: str, payload: TranscribeJobRequest) -> None:
    logger.info('Starting job %s', job_id)
    try:
        store.update_job(job_id, status='running')
        exit_code, stdout_tail, stderr_tail = run_pipeline_job(settings, payload)
        if exit_code == 0:
            store.update_job(
                job_id,
                status='completed',
                result_message='Pipeline completed successfully',
                exit_code=exit_code,
                stdout_tail=stdout_tail or None,
                stderr_tail=stderr_tail or None,
            )
            logger.info('Job completed %s', job_id)
            return

        store.update_job(
            job_id,
            status='failed',
            error_message='Pipeline process exited with non-zero code',
            exit_code=exit_code,
            stdout_tail=stdout_tail or None,
            stderr_tail=stderr_tail or None,
        )
        logger.error('Job failed %s with exit_code=%s', job_id, exit_code)
    except Exception as error:
        store.update_job(
            job_id,
            status='failed',
            error_message=f'Unhandled job error: {error}',
        )
        logger.exception('Unhandled failure for job %s', job_id)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
