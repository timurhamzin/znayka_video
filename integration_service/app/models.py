from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal['queued', 'running', 'completed', 'failed']


class TranscribeJobRequest(BaseModel):
    video_name: str | None = None
    whisper_language: str | None = None
    translation_source_language: str | None = None
    translation_target_language: str | None = None
    offline_mode: bool | None = None
    dry_run: bool = False


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    request: TranscribeJobRequest
    result_message: str | None = None
    error_message: str | None = None
    exit_code: int | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None


class JobCreatedResponse(BaseModel):
    id: str = Field(..., description='Created job identifier.')
    status: JobStatus
