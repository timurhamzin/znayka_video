from __future__ import annotations

import os
import subprocess
import sys

from .config import Settings
from .models import TranscribeJobRequest

TAIL_LIMIT = 4000


def run_pipeline_job(
    settings: Settings,
    request: TranscribeJobRequest,
) -> tuple[int, str, str]:
    if request.dry_run:
        return 0, 'dry_run enabled: pipeline execution skipped', ''

    env = os.environ.copy()
    env.update(_build_pipeline_env(request))

    command = [sys.executable, str(settings.pipeline_entrypoint)]
    process = subprocess.run(
        command,
        cwd=settings.repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_tail = process.stdout[-TAIL_LIMIT:]
    stderr_tail = process.stderr[-TAIL_LIMIT:]
    return process.returncode, stdout_tail, stderr_tail


def _build_pipeline_env(request: TranscribeJobRequest) -> dict[str, str]:
    env_updates = {
        'TRANSCRIBE_INTERACTIVE': 'false',
        'TRANSCRIBE_RUN_GENERATE_SPANS': 'false',
        'TRANSCRIBE_RUN_FILTER_SIDECARS': 'false',
        'TRANSCRIBE_RUN_TRANSCRIPTION': 'true',
        'TRANSCRIBE_RUN_TRANSLATION': 'true',
        'TRANSCRIBE_RUN_MERGE': 'false',
        'TRANSCRIBE_RUN_SIDECAR_REPLACE': 'false',
        'TRANSCRIBE_RUN_BAKE_SUBTITLES': 'false',
    }

    if request.video_name:
        env_updates['TRANSCRIBE_SINGLE_VIDEO'] = request.video_name
    if request.whisper_language:
        env_updates['TRANSCRIBE_WHISPER_LANGUAGE'] = request.whisper_language
    if request.translation_source_language:
        env_updates['TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE'] = (
            request.translation_source_language
        )
    if request.translation_target_language:
        env_updates['TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE'] = (
            request.translation_target_language
        )
    if request.offline_mode is not None:
        env_updates['TRANSCRIBE_OFFLINE_MODE'] = (
            'true' if request.offline_mode else 'false'
        )

    return env_updates
