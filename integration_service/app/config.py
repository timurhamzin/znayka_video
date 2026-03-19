from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    jobs_file: Path
    pipeline_entrypoint: Path
    root_env_file: Path


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    service_root = repo_root / 'integration_service'

    load_dotenv(service_root / '.env', override=False)

    jobs_file_env = os.getenv(
        'INTEGRATION_JOBS_FILE',
        str(service_root / 'data' / 'jobs.json'),
    )
    pipeline_entrypoint_env = os.getenv(
        'INTEGRATION_PIPELINE_ENTRYPOINT',
        'main.py',
    )
    root_env_file_env = os.getenv('INTEGRATION_ROOT_ENV_FILE', '.env')

    jobs_file = Path(jobs_file_env)
    if not jobs_file.is_absolute():
        jobs_file = repo_root / jobs_file

    pipeline_entrypoint = Path(pipeline_entrypoint_env)
    if not pipeline_entrypoint.is_absolute():
        pipeline_entrypoint = repo_root / pipeline_entrypoint

    root_env_file = Path(root_env_file_env)
    if not root_env_file.is_absolute():
        root_env_file = repo_root / root_env_file

    return Settings(
        repo_root=repo_root,
        jobs_file=jobs_file,
        pipeline_entrypoint=pipeline_entrypoint,
        root_env_file=root_env_file,
    )
