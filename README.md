# Setup

## Prerequisites

1. Install Python dependencies:

```bash
uv sync
```

If environment is stale:

```bash
uv sync --reinstall
```

2. (Windows) Install Node.js for downloader support:

```powershell
winget install OpenJS.NodeJS
```

3. Optional NLTK data for English phonetic respelling:

```bash
python -m nltk.downloader averaged_perceptron_tagger cmudict punkt
```

## Environment

Copy `.env.example` to `.env` and edit the `TRANSCRIBE_*` variables.

Required core variables:

```env
TRANSCRIBE_VIDEO_FOLDER=path/to/folder/with/mp4
TRANSCRIBE_WHISPER_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE=en
TRANSCRIBE_SIDECAR_SRT_ENCODING=utf-8
TRANSCRIBE_HF_TOKEN=
TRANSCRIBE_OFFLINE_MODE=false
```

Master step flags (`main.py`):

```env
TRANSCRIBE_RUN_GENERATE_SPANS=true
TRANSCRIBE_RUN_FILTER_SIDECARS=true
TRANSCRIBE_RUN_TRANSCRIPTION=false
TRANSCRIBE_RUN_TRANSLATION=false
TRANSCRIBE_RUN_SIDECAR_REPLACE=false
TRANSCRIBE_RUN_MERGE=true
```

Force flags:

```env
TRANSCRIBE_FORCE_SPANS=true
TRANSCRIBE_FORCE_TRANSCRIPTION=false
TRANSCRIBE_FORCE_TRANSLATION=true
```

Notes:
- Translation is independent from transcription.
- `TRANSCRIBE_RUN_TRANSLATION=true` runs translation from sidecar (`video_name.srt`) via `TRANSCRIBE_TRANSLATION_INPUT=sidecar`.
- Sidecar replacement variant is derived from `TRANSCRIBE_SIDECAR_SRT_ENCODING`:
  - `utf-8` -> `translated_utf8`
  - `windows-1251`/`cp1251` -> `translated_windows1251`
- Merge always uses video folder as source and merges translated files for the target encoding variant.
- If a prerequisite is missing, `main.py` asks interactively and remembers your answer in `.master_pipeline_memory.json`.
- For single-video testing, set `TRANSCRIBE_SINGLE_VIDEO=Exact Video Name.mp4` (or stem).

## Run

Windows:

```powershell
.\scripts\windows\run_main.ps1
```

Linux/macOS:

```bash
chmod +x ./scripts/unix/run_main.sh
./scripts/unix/run_main.sh
```

## Master Profiles

`main.py` contains commented profile blocks at the top. They are ready-to-copy into `.env` and differ only by step flags.

Profiles include:
- force create spans
- force filter sidecars
- force rerun translation from sidecar
- force rerun transcription + translation
- all-in-one with merge

## Output Layout

Per video:

- `Video Name.mp4`
- `Video Name.srt` (sidecar)
- `Video Name.speech_spans.json`
- `Video Name/original/Video Name.srt`
- `Video Name/translated_utf8/Video Name.srt`
- `Video Name/translated_windows1251/Video Name.srt`

Merge output:
- `TRANSCRIBE_SUBTITLE_OUTPUT_MD` (default: `<video_folder>/merged_srt_files.md`)

## Scripts

- `main.py` is the single orchestrator for spans, filtering, transcription, translation, merge, and optional sidecar replacement.
- `transcribe.py` provides the underlying transcription/translation/speech-span pipeline.
- `sidecar_replace.py` is used by `main.py` for the sidecar replacement step.
- `subtitles_to_markdown.py` is used by `main.py` for merge output generation.
