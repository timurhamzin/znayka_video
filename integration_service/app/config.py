from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from arq.connections import RedisSettings
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    pipeline_entrypoint: Path
    redis_host: str
    redis_port: int
    redis_database: int
    redis_password: str | None
    app_host: str
    app_port: int


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    service_root = repo_root / 'integration_service'

    load_dotenv(service_root / '.env', override=False)

    pipeline_entrypoint_env = os.getenv(
        'INTEGRATION_PIPELINE_ENTRYPOINT',
        'main.py',
    )
    redis_host = os.getenv('INTEGRATION_REDIS_HOST', '127.0.0.1')
    redis_port = int(os.getenv('INTEGRATION_REDIS_PORT', '6379'))
    redis_database = int(os.getenv('INTEGRATION_REDIS_DB', '0'))
    redis_password = os.getenv('INTEGRATION_REDIS_PASSWORD') or None
    app_host = os.getenv('INTEGRATION_APP_HOST', '127.0.0.1')
    app_port = int(os.getenv('INTEGRATION_APP_PORT', '8010'))

    pipeline_entrypoint = Path(pipeline_entrypoint_env)
    if not pipeline_entrypoint.is_absolute():
        pipeline_entrypoint = repo_root / pipeline_entrypoint

    return Settings(
        repo_root=repo_root,
        pipeline_entrypoint=pipeline_entrypoint,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_database=redis_database,
        redis_password=redis_password,
        app_host=app_host,
        app_port=app_port,
    )


def to_redis_settings(settings: Settings) -> RedisSettings:
    return RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_database,
        password=settings.redis_password,
    )
