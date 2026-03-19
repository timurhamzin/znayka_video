from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import JobRecord


class JobStore:
    def __init__(self, jobs_file: Path) -> None:
        self._jobs_file = jobs_file
        self._lock = threading.Lock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self._jobs_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._jobs_file.exists():
            self._jobs_file.write_text('[]', encoding='utf-8')

    def _read_jobs(self) -> list[dict]:
        if not self._jobs_file.exists():
            return []
        raw = self._jobs_file.read_text(encoding='utf-8-sig')
        if not raw.strip():
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    def _write_jobs(self, jobs: list[dict]) -> None:
        tmp_file = self._jobs_file.with_suffix('.tmp')
        payload = json.dumps(jobs, ensure_ascii=True, indent=2)
        tmp_file.write_text(payload, encoding='utf-8')
        tmp_file.replace(self._jobs_file)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            jobs = [JobRecord.model_validate(item) for item in self._read_jobs()]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            for item in self._read_jobs():
                if item.get('id') == job_id:
                    return JobRecord.model_validate(item)
        return None

    def add_job(self, job: JobRecord) -> None:
        with self._lock:
            jobs = self._read_jobs()
            jobs.append(job.model_dump(mode='json'))
            self._write_jobs(jobs)

    def update_job(self, job_id: str, **fields: object) -> JobRecord:
        with self._lock:
            jobs = self._read_jobs()
            for index, item in enumerate(jobs):
                if item.get('id') != job_id:
                    continue
                item.update(fields)
                item['updated_at'] = _now_utc().isoformat()
                jobs[index] = item
                self._write_jobs(jobs)
                return JobRecord.model_validate(item)
        raise KeyError(f'Job not found: {job_id}')


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
