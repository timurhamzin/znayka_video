from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal['queued', 'running', 'completed', 'failed']
ExplicitCutPlanStatus = Literal['planned', 'approved', 'rejected', 'applied']


class TranscribeJobRequest(BaseModel):
    video_name: str | None = None
    whisper_language: str | None = None
    translation_source_language: str | None = None
    translation_target_language: str | None = None
    offline_mode: bool | None = None
    dry_run: bool = False
    run_generate_spans: bool = False
    run_filter_sidecars: bool = False
    run_transcription: bool = True
    run_translation: bool = True
    run_merge: bool = False
    run_sidecar_replace: bool = False
    run_bake_subtitles: bool = False


class ExplicitCutPlanRequest(BaseModel):
    video_name: str = Field(..., description='Target video filename or stem.')
    frame_verification_backend: str | None = Field(
        default=None,
        description='Optional override for explicit-cut frame verification backend.',
    )
    frame_interval_sec: float | None = None
    frame_nsfw_threshold: float | None = None
    frame_min_positive_ratio: float | None = None
    force_replan: bool = False


class ExplicitCutPlanDecisionRequest(BaseModel):
    approved: bool = Field(..., description='Whether the generated cut plan is approved.')
    note: str | None = None


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


class ExplicitCutPlanRecord(BaseModel):
    id: str
    status: ExplicitCutPlanStatus
    created_at: datetime
    updated_at: datetime
    request: ExplicitCutPlanRequest
    plan: dict[str, Any]
    note: str | None = None
    result_message: str | None = None


class JobCreatedResponse(BaseModel):
    id: str = Field(..., description='Created job identifier.')
    status: JobStatus


class ExplicitCutPlanCreatedResponse(BaseModel):
    id: str = Field(..., description='Created explicit-cut plan identifier.')
    status: ExplicitCutPlanStatus
