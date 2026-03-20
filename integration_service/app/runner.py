from __future__ import annotations

import asyncio
import os

from .config import Settings
from .models import TranscribeJobRequest

TAIL_LIMIT = 4000


async def run_pipeline_job(
    settings: Settings,
    request: TranscribeJobRequest,
) -> tuple[int, str, str]:
    if request.dry_run:
        return 0, 'dry_run enabled: pipeline execution skipped', ''

    env = os.environ.copy()
    env.update(_build_pipeline_env(request))

    command = [
        'uv',
        'run',
        '--project',
        str(settings.repo_root),
        'python',
        str(settings.pipeline_entrypoint),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=settings.repo_root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    stdout_tail = stdout_bytes.decode(errors='replace')[-TAIL_LIMIT:]
    stderr_tail = stderr_bytes.decode(errors='replace')[-TAIL_LIMIT:]
    return process.returncode or 0, stdout_tail, stderr_tail


def _build_pipeline_env(request: TranscribeJobRequest) -> dict[str, str]:
    env_updates = {
        'TRANSCRIBE_INTERACTIVE': 'false',
        'TRANSCRIBE_RUN_GENERATE_SPANS': _flag(request.run_generate_spans),
        'TRANSCRIBE_RUN_FILTER_SIDECARS': _flag(request.run_filter_sidecars),
        'TRANSCRIBE_RUN_TRANSCRIPTION': _flag(request.run_transcription),
        'TRANSCRIBE_RUN_TRANSLATION': _flag(request.run_translation),
        'TRANSCRIBE_RUN_MERGE': _flag(request.run_merge),
        'TRANSCRIBE_RUN_SIDECAR_REPLACE': _flag(request.run_sidecar_replace),
        'TRANSCRIBE_RUN_BAKE_SUBTITLES': _flag(request.run_bake_subtitles),
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


def _flag(value: bool) -> str:
    return 'true' if value else 'false'
