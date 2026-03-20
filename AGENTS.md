IMPORTANT: When applicable, prefer using pycharm-index MCP tools (if available) for code navigation and refactoring.

## Project Rules
- `.qwen/rules/python-codestyle.md` — Python code style and best practices

## Transcribe Environment Variables
- Use explicit `transcribe.py` variables grouped together in `.env`:
  - `TRANSCRIBE_VIDEO_FOLDER`
  - `TRANSCRIBE_WHISPER_LANGUAGE`
  - `TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE`
  - `TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE`
  - `TRANSCRIBE_TRANSLATION_MODEL`
  - `TRANSCRIBE_ENABLE_PHONETIC`
  - `TRANSCRIBE_PHONETIC_LANGUAGE`
  - `TRANSCRIBE_SIDECAR_SRT_ENCODING`
  - `TRANSCRIBE_HF_TOKEN`
  - `TRANSCRIBE_OFFLINE_MODE`
  - `TRANSCRIBE_OUTPUT_FOLDER`
- Keep legacy variables only for backward compatibility (`TRANSCRIBE_DUPLICATE_SRT_ENCODING`, `VIDEO_FOLDER`, `LANGUAGE`, `DUPLICATE_SRT_ENCODING`, `HF_TOKEN`, `OUTPUT_FOLDER`).
- For French input translated to English, set:
  - `TRANSCRIBE_WHISPER_LANGUAGE=fr`
  - `TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE=fr`
  - `TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE=en`
- English phonetic respelling is only supported when source language is English.
- Keep `TRANSCRIBE_OFFLINE_MODE=false` for first run so models can be downloaded.

## Run Scripts
- Windows PowerShell:
  - `.\run_download.ps1`
  - `.\run_transcribe.ps1`
- Linux/macOS bash:
  - `./run_download.sh`
  - `./run_transcribe.sh`

## Dependency Installation
- Install/update env with `uv sync`.
- If runtime errors mention missing tokenizer/model deps, run `uv sync --reinstall`.
- `transcribe.py` requires `sentencepiece` and `protobuf` for Marian tokenizer/model loading.

## Integration Plan
- Follow and update the checklist in `docs/integration-plan.md`.
- Keep integration implementation in `integration_service/` as a separate project
  until monorepo migration.
- Use `integration_service/scripts/local/run_app.py` (app) and
  `integration_service/scripts/local/run_worker.py` (worker) for local/PyCharm
  runs.
