from __future__ import annotations

from datetime import datetime, timezone

from arq.connections import ArqRedis

from .models import JobRecord

JOBS_INDEX_KEY = 'integration:jobs:index'
JOB_KEY_PREFIX = 'integration:jobs:'


class RedisJobStore:
    def __init__(self, redis: ArqRedis) -> None:
        self._redis = redis

    async def add_job(self, job: JobRecord) -> None:
        job_key = _job_key(job.id)
        payload = job.model_dump_json()
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.set(job_key, payload)
            pipeline.lpush(JOBS_INDEX_KEY, job.id)
            await pipeline.execute()

    async def get_job(self, job_id: str) -> JobRecord | None:
        raw = await self._redis.get(_job_key(job_id))
        if raw is None:
            return None
        return JobRecord.model_validate_json(raw)

    async def list_jobs(self, limit: int = 100) -> list[JobRecord]:
        ids = await self._redis.lrange(JOBS_INDEX_KEY, 0, max(limit - 1, 0))
        result: list[JobRecord] = []
        for job_id in ids:
            job = await self.get_job(job_id)
            if job is not None:
                result.append(job)
        return result

    async def update_job(self, job_id: str, **fields: object) -> JobRecord:
        current = await self.get_job(job_id)
        if current is None:
            raise KeyError(f'Job not found: {job_id}')

        updated_payload = current.model_dump(mode='json')
        updated_payload.update(fields)
        updated_payload['updated_at'] = _now_utc().isoformat()
        updated = JobRecord.model_validate(updated_payload)
        await self._redis.set(_job_key(job_id), updated.model_dump_json())
        return updated


def _job_key(job_id: str) -> str:
    return f'{JOB_KEY_PREFIX}{job_id}'


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
