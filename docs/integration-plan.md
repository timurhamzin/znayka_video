# Integration Plan: `znayka-video` -> Service Module

This plan integrates the current script bundle into a bigger service architecture
without rewriting core transcription logic.

## Goal

- Keep current scripts operational.
- Add a separate API project that can run transcription jobs.
- Prepare a clean migration path into the future monorepo backend.

## Scope

- In scope:
  - Separate FastAPI integration service in this repository.
  - Job API with Redis-backed queue and state for MVP.
  - Background execution that calls existing pipeline scripts.
  - Documentation and operational checklist.
- Out of scope (for now):
  - Authentication/authorization.
  - Production-grade persistence (PostgreSQL) in this repository.

## Phased Checklist

### Phase 0: Planning and Alignment

- [x] Define architecture: Expo client -> API -> worker runner.
- [x] Keep `znayka-video` scripts as source of truth for processing logic.
- [x] Decide no full rewrite before first service integration.

Acceptance criteria:
- [x] Written implementation plan exists and is linked from project docs.

### Phase 1: Separate Integration Service (Current Repository)

- [x] Create `integration_service/` as standalone project.
- [x] Add service-level `README.md` with run instructions.
- [x] Add service-level `.env.example`.
- [x] Add service-level `pyproject.toml`.
- [x] Implement FastAPI app with health endpoint.
- [x] Add basic HTML control form with pipeline step tickboxes.
- [x] Implement job API:
  - [x] `POST /jobs/transcribe`
  - [x] `GET /jobs/{job_id}`
  - [x] `GET /jobs`
- [x] Implement Redis-based queue and job state store.
- [x] Implement ARQ worker as separate process.
- [x] Implement async runner that executes existing root pipeline.
- [x] Return structured status transitions (`queued`, `running`, `completed`,
  `failed`).
- [x] Add `docker-compose` for app + worker + redis.

Acceptance criteria:
- [x] Can create a job and observe status transitions end-to-end.
- [ ] Failed pipeline run marks job as `failed` with error details.

### Phase 2: Config and Contract Hardening

- [ ] Map job request fields to explicit `TRANSCRIBE_*` env variables.
- [ ] Keep legacy env vars as compatibility fallback only.
- [ ] Add API request/response schema documentation.
- [ ] Add idempotency strategy for duplicate submissions (MVP key or hash).
- [ ] Add basic validation for video selection and folder existence.

Acceptance criteria:
- [ ] Input contract documented and stable.
- [ ] Invalid payloads rejected with clear errors.

### Phase 3: Packaging for Monorepo Integration

- [ ] Extract reusable pipeline entrypoint in Python module (callable, not only
  CLI).
- [ ] Replace subprocess bridge with direct function call where possible.
- [ ] Add adapter layer so monorepo backend can import and trigger jobs.
- [ ] Keep CLI wrappers for manual/local runs.

Acceptance criteria:
- [ ] Integration backend can trigger jobs without shelling out.

### Phase 4: Production Readiness

- [ ] Replace Redis-only job history with PostgreSQL-backed persistence.
- [x] Add queue worker (ARQ) for concurrent jobs.
- [ ] Add retries and timeout policies.
- [ ] Add observability:
  - [ ] structured logs
  - [ ] request/job correlation IDs
  - [ ] metrics (duration, success/failure counts)

Acceptance criteria:
- [ ] Multiple concurrent jobs can be processed reliably.
- [ ] Job history persists across restarts.

### Phase 5: Expo Integration

- [ ] Add frontend API client methods:
  - [ ] create job
  - [ ] list jobs
  - [ ] poll job status
- [ ] Add UI states in Expo:
  - [ ] queued/running/completed/failed
  - [ ] error message rendering
  - [ ] result path visibility
- [ ] Add environment-specific base URL guidance for web/emulator/device.

Acceptance criteria:
- [ ] Job can be started from Expo and tracked to completion.

## Verification Checklist

- [x] `uv sync` succeeds in `integration_service/`.
- [ ] `uv run uvicorn app.main:app --reload --port 8010` starts service.
- [ ] `GET /health` returns ok response.
- [x] `POST /jobs/transcribe` creates a job.
- [x] `GET /jobs/{job_id}` reflects state changes.
- [ ] Root scripts remain runnable through existing script entrypoints.

## Risks and Mitigations

- Risk: subprocess integration can drift from script CLI behavior.
  - Mitigation: keep a single command path and add integration tests.
- Risk: Redis state can be lost without persistence config.
  - Mitigation: enable Redis persistence for non-dev and move to PostgreSQL in
    Phase 4.
- Risk: long-running jobs block process resources.
  - Mitigation: move to dedicated queue worker in Phase 4.

## Change Log

- 2026-03-20:
  - Plan created.
  - Phase 1 implementation started.
  - Separate `integration_service/` scaffold created.
  - Job API and Redis-backed job store implemented.
  - ARQ async worker implemented as separate process.
  - HTML form with tickboxes added for job submission.
  - Docker Compose stack added (`app`, `worker`, `redis`).
  - Local runtime folders moved under ignored `local_testing/`.
  - Root `.dockerignore` aligned with `.gitignore`.
  - `dry_run` execution mode added for safe lifecycle verification.
