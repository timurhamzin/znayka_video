from __future__ import annotations

import logging
from typing import Any

from arq.connections import ArqRedis

from .config import Settings, load_settings, to_redis_settings
from .models import TranscribeJobRequest
from .runner import run_pipeline_job
from .store import RedisJobStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


async def startup(ctx: dict[str, Any]) -> None:
    settings = load_settings()
    ctx['settings'] = settings


async def run_pipeline_job_task(
    ctx: dict[str, Any],
    job_id: str,
    payload: dict[str, Any],
) -> None:
    settings: Settings = ctx['settings']
    redis: ArqRedis = ctx['redis']
    store = RedisJobStore(redis)
    request = TranscribeJobRequest.model_validate(payload)

    logger.info('Worker picked job %s', job_id)
    try:
        await store.update_job(job_id, status='running')
        exit_code, stdout_tail, stderr_tail = await run_pipeline_job(settings, request)
        if exit_code == 0:
            await store.update_job(
                job_id,
                status='completed',
                result_message='Pipeline completed successfully',
                exit_code=exit_code,
                stdout_tail=stdout_tail or None,
                stderr_tail=stderr_tail or None,
            )
            logger.info('Worker completed job %s', job_id)
            return

        await store.update_job(
            job_id,
            status='failed',
            error_message='Pipeline process exited with non-zero code',
            exit_code=exit_code,
            stdout_tail=stdout_tail or None,
            stderr_tail=stderr_tail or None,
        )
        logger.error('Worker failed job %s with exit_code=%s', job_id, exit_code)
    except Exception as error:
        await store.update_job(
            job_id,
            status='failed',
            error_message=f'Unhandled worker error: {error}',
        )
        logger.exception('Worker unhandled failure for job %s', job_id)


class WorkerSettings:
    functions = [run_pipeline_job_task]
    on_startup = startup
    redis_settings = to_redis_settings(load_settings())
    max_jobs = 2
    job_timeout = 60 * 60 * 12
