from __future__ import annotations

import os
from pathlib import Path

from explicit_content_cut import (
    ExplicitCutConfig,
    ExplicitCutPlan,
    apply_explicit_cut_plan,
    build_explicit_cut_plan,
    load_explicit_cut_config_from_env,
)

from .config import Settings
from .models import ExplicitCutPlanRequest


def resolve_video_path(settings: Settings, video_name: str) -> Path:
    video_folder_raw = os.getenv('TRANSCRIBE_VIDEO_FOLDER') or os.getenv('VIDEO_FOLDER')
    if not video_folder_raw:
        raise RuntimeError('TRANSCRIBE_VIDEO_FOLDER (or VIDEO_FOLDER) is required.')

    video_folder = Path(video_folder_raw)
    if not video_folder.is_absolute():
        video_folder = settings.repo_root / video_folder
    if not video_folder.exists():
        raise RuntimeError(f'Video folder does not exist: {video_folder}')

    normalized = video_name.strip().lower()
    videos = sorted(video_folder.glob('*.mp4'))
    for video in videos:
        if video.name.lower() == normalized or video.stem.lower() == normalized:
            return video
    raise FileNotFoundError(f'Video not found: {video_name}')


def build_config(request: ExplicitCutPlanRequest) -> ExplicitCutConfig:
    base = load_explicit_cut_config_from_env()
    return ExplicitCutConfig(
        sidecar_encoding=base.sidecar_encoding,
        keywords=base.keywords,
        margin_before_sec=base.margin_before_sec,
        margin_after_sec=base.margin_after_sec,
        group_gap_sec=base.group_gap_sec,
        scene_gap_sec=base.scene_gap_sec,
        max_extend_before_sec=base.max_extend_before_sec,
        max_extend_after_sec=base.max_extend_after_sec,
        preset=base.preset,
        crf=base.crf,
        frame_verification_backend=(
            request.frame_verification_backend.strip().lower()
            if request.frame_verification_backend
            else base.frame_verification_backend
        ),
        frame_interval_sec=(
            request.frame_interval_sec
            if request.frame_interval_sec is not None
            else base.frame_interval_sec
        ),
        frame_nsfw_threshold=(
            request.frame_nsfw_threshold
            if request.frame_nsfw_threshold is not None
            else base.frame_nsfw_threshold
        ),
        frame_min_positive_ratio=(
            request.frame_min_positive_ratio
            if request.frame_min_positive_ratio is not None
            else base.frame_min_positive_ratio
        ),
        frame_max_samples_per_span=base.frame_max_samples_per_span,
        force=base.force or request.force_replan,
    )


def create_plan(settings: Settings, request: ExplicitCutPlanRequest) -> ExplicitCutPlan:
    video = resolve_video_path(settings, request.video_name)
    config = build_config(request)
    return build_explicit_cut_plan(video, config)


def apply_plan(
    settings: Settings,
    request: ExplicitCutPlanRequest,
    plan: ExplicitCutPlan,
) -> ExplicitCutPlan:
    video = resolve_video_path(settings, request.video_name)
    config = build_config(request)
    return apply_explicit_cut_plan(video, plan, config)
