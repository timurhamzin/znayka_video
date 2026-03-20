FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

# Copy dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock ./
COPY integration_service/pyproject.toml integration_service/uv.lock integration_service/

# Root project dependencies are required by the pipeline worker (main.py/transcribe.py).
RUN uv sync --frozen
RUN uv sync --project integration_service --frozen

COPY . .

ENV PYTHONUNBUFFERED=1
