# Setup

## Windows Prerequisites

Install Node.js runtime (required for YouTube extraction):

```powershell
winget install OpenJS.NodeJS
```

## Install Dependencies

```bash
uv sync
```

If you already had an older environment, refresh it:

```bash
uv sync --reinstall
```

## Download NLTK Data (Required for English phonetic respelling)

Run once after installation:

```bash
python -m nltk.downloader averaged_perceptron_tagger cmudict punkt
```

## Configure Environment

Copy `.env.example` to `.env` and edit these variables.

### Downloader vars

```env
DOWNLOAD_URL=https://example.com/video-or-playlist
VIDEO_RESOLUTION=720p
```

### `transcribe.py` vars (keep together)

```env
TRANSCRIBE_VIDEO_FOLDER=path/to/folder/with/mp4
TRANSCRIBE_WHISPER_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE=en
TRANSCRIBE_TRANSLATION_MODEL=
TRANSCRIBE_ENABLE_PHONETIC=true
TRANSCRIBE_PHONETIC_LANGUAGE=en
TRANSCRIBE_SIDECAR_SRT_ENCODING=utf-8
TRANSCRIBE_ENABLE_SPEECH_SPANS=true
TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING=true
TRANSCRIBE_SPEECH_SPANS_OVERWRITE=false
TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER=true
TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS=true
TRANSCRIBE_SPEECH_SPANS_ONLY_MODE=false
TRANSCRIBE_TRANSLATION_ONLY_MODE=false
TRANSCRIBE_TRANSLATION_INPUT=original
TRANSCRIBE_TRANSLATION_OVERWRITE=false
TRANSCRIBE_TRANSLATION_APPEND_SOURCE=true
TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION=false
TRANSCRIBE_SPEECH_SPAN_MARGIN_SEC=0.35
TRANSCRIBE_SPEECH_VAD_THRESHOLD=0.3
TRANSCRIBE_SPEECH_MIN_DURATION_SEC=0.15
TRANSCRIBE_SPEECH_MIN_SILENCE_SEC=0.2
TRANSCRIBE_SPEECH_SPAN_FILE_SUFFIX=.speech_spans.json
TRANSCRIBE_SPEECH_FILTER_RESCUE_DENSE_RUNS=true
TRANSCRIBE_SPEECH_FILTER_RESCUE_MIN_CUES=4
TRANSCRIBE_SPEECH_FILTER_RESCUE_MIN_DURATION_SEC=8
TRANSCRIBE_SPEECH_FILTER_RESCUE_MAX_DURATION_SEC=90
TRANSCRIBE_SPEECH_FILTER_RESCUE_MIN_DENSITY=0.08
TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_RUNS=true
TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_MAX_CUES=3
TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_MAX_DURATION_SEC=4
TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_NEIGHBOR_GAP_SEC=8
TRANSCRIBE_SPEECH_FILTER_RESCUE_BRIDGE_RUNS=true
TRANSCRIBE_SPEECH_FILTER_RESCUE_BRIDGE_MAX_CUES=6
TRANSCRIBE_SPEECH_FILTER_RESCUE_BRIDGE_MAX_DURATION_SEC=25
TRANSCRIBE_SIDECAR_REPLACE_VARIANT=translated_utf8
TRANSCRIBE_SINGLE_VIDEO=
TRANSCRIBE_HF_TOKEN=your_hf_token
TRANSCRIBE_OFFLINE_MODE=false
TRANSCRIBE_OUTPUT_FOLDER=
```

Notes:
- If `TRANSCRIBE_TRANSLATION_MODEL` is empty, model is auto-selected from source/target language pair.
- English phonetic respelling works only when source language is English. For French source, phonetic lines are skipped.
- Keep `TRANSCRIBE_OFFLINE_MODE=false` for first run so Marian models can be downloaded.
- Speech spans are detected with a local Silero VAD model.
- Sidecar `video_name.srt` can be auto-filtered to keep only cues that overlap speech spans.
- Dense subtitle runs in the middle of kept speech regions are rescued to avoid false VAD-hole deletions.
- Tiny middle runs between two nearby kept regions are also rescued (for short interjections).
- If `video_name.speech_spans.json` already exists, it is reused (unless overwrite is enabled).
- `TRANSCRIBE_SPEECH_SPANS_ONLY_MODE=true` runs only span detection + sidecar filtering (no Whisper/translation).
- `TRANSCRIBE_TRANSLATION_ONLY_MODE=true` runs translation without Whisper.
- `TRANSCRIBE_TRANSLATION_INPUT=sidecar` translates from `video_name.srt` next to video.
- `TRANSCRIBE_TRANSLATION_INPUT=original` translates from `video_name/original/video_name.srt`.
- `TRANSCRIBE_TRANSLATION_APPEND_SOURCE` controls whether translated files include source lines.
- `TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION` controls whether main pipeline overwrites sidecar from translation.
- `TRANSCRIBE_SIDECAR_REPLACE_VARIANT` is used by `sidecar_replace.py` (`original`, `translated_utf8`, `translated_windows1251`).
- `TRANSCRIBE_SINGLE_VIDEO` lets you test one file by exact filename (or stem) without processing the whole folder.
- Legacy vars are still supported for backward compatibility:
  - `TRANSCRIBE_DUPLICATE_SRT_ENCODING`
  - `VIDEO_FOLDER`
  - `LANGUAGE`
  - `DUPLICATE_SRT_ENCODING`
  - `HF_TOKEN`
  - `OUTPUT_FOLDER`

### `subtitles_to_markdown.py` vars

```env
TRANSCRIBE_SUBTITLE_SOURCE_DIR=path/to/folder/with/srt
TRANSCRIBE_SUBTITLE_OUTPUT_MD=path/to/output/subtitles.md
TRANSCRIBE_SUBTITLE_GLOB=*.srt
TRANSCRIBE_SUBTITLE_SOURCE_ENCODING=
TRANSCRIBE_SUBTITLE_SOURCE_ENCODING_FALLBACKS=utf-8,utf-8-sig,windows-1251,cp1251
TRANSCRIBE_SUBTITLE_OUTPUT_ENCODING=
```

Notes:
- `TRANSCRIBE_SUBTITLE_SOURCE_ENCODING` controls how `.srt` files are read.
- `TRANSCRIBE_SUBTITLE_SOURCE_ENCODING_FALLBACKS` is used when the primary source encoding fails (useful for mixed-encoding folders).
- If `TRANSCRIBE_SUBTITLE_OUTPUT_ENCODING` is not set, output defaults to `TRANSCRIBE_SIDECAR_SRT_ENCODING` (and then legacy names).

## Run Scripts (activate `.venv` automatically)

### Windows (PowerShell)

```powershell
# Fast scripts (no Whisper):
.\scripts\windows\fast\run_download.ps1
.\scripts\windows\fast\run_subtitles_to_markdown.ps1
.\scripts\windows\fast\run_speech_spans_generate.ps1
.\scripts\windows\fast\run_speech_spans_apply.ps1
.\scripts\windows\fast\run_speech_force_create_spans.ps1
.\scripts\windows\fast\run_speech_force_filter_sidecars.ps1
.\scripts\windows\fast\run_speech_force_rerun_translation_from_sidecars.ps1
.\scripts\windows\fast\run_sidecar_replace.ps1

# Whisper scripts (long-running):
.\scripts\windows\whisper\run_transcribe.ps1
.\scripts\windows\whisper\run_speech_rerun_translation.ps1
.\scripts\windows\whisper\run_speech_all_in_one.ps1
.\scripts\windows\whisper\run_speech_force_rerun_transcribe_translate.ps1
```

### Linux/macOS (bash)

```bash
chmod +x scripts/unix/fast/*.sh scripts/unix/whisper/*.sh
./scripts/unix/fast/run_download.sh
./scripts/unix/fast/run_subtitles_to_markdown.sh
./scripts/unix/fast/run_speech_spans_generate.sh
./scripts/unix/fast/run_speech_spans_apply.sh
./scripts/unix/fast/run_speech_force_create_spans.sh
./scripts/unix/fast/run_speech_force_filter_sidecars.sh
./scripts/unix/fast/run_speech_force_rerun_translation_from_sidecars.sh
./scripts/unix/fast/run_sidecar_replace.sh
./scripts/unix/whisper/run_transcribe.sh
./scripts/unix/whisper/run_speech_rerun_translation.sh
./scripts/unix/whisper/run_speech_all_in_one.sh
./scripts/unix/whisper/run_speech_force_rerun_transcribe_translate.sh
```

## French Video -> English Translation Example

Set:

```env
TRANSCRIBE_WHISPER_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE=en
```

This will transcribe French audio and translate subtitles to English.

## Common Speech-Span Workflows

Set your target folder:

```powershell
$env:TRANSCRIBE_VIDEO_FOLDER='C:\Timur\work\znayka-video\processed\Bebebears - TV cartoons'
```

### 1) Create `*.speech_spans.json` only

```powershell
.\scripts\windows\fast\run_speech_spans_generate.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
# optional:
# .\scripts\windows\fast\run_speech_spans_generate.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER -SingleVideo "Some Episode.mp4" -OverwriteSpans true
```

Force version (always rebuild spans):

```powershell
.\scripts\windows\fast\run_speech_force_create_spans.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
```

### 2) Apply spans to already existing sidecar `.srt` files

```powershell
.\scripts\windows\fast\run_speech_spans_apply.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
# optional:
# .\scripts\windows\fast\run_speech_spans_apply.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER -SingleVideo "Some Episode.mp4"
```

Force version (rebuild spans + re-filter sidecars):

```powershell
.\scripts\windows\fast\run_speech_force_filter_sidecars.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
```

### 3) Rerun translation after filtering

Use Whisper-coupled rerun:

```powershell
.\scripts\windows\whisper\run_speech_rerun_translation.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
# optional:
# .\scripts\windows\whisper\run_speech_rerun_translation.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER -SingleVideo "Some Episode.mp4"
```

Force version (delete per-video output folders, then rerun transcription + translation):

```powershell
.\scripts\windows\whisper\run_speech_force_rerun_transcribe_translate.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
```

Force translation-only from filtered sidecars (no Whisper):

```powershell
.\scripts\windows\fast\run_speech_force_rerun_translation_from_sidecars.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
```

Replace sidecar from a chosen variant folder:

```powershell
.\scripts\windows\fast\run_sidecar_replace.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER -Variant original
# Variant values: original | translated_utf8 | translated_windows1251
```

### All at once (spans + filter + span-aware Whisper + translation)

Yes, this is supported in one run:

```powershell
.\scripts\windows\whisper\run_speech_all_in_one.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER
# optional:
# .\scripts\windows\whisper\run_speech_all_in_one.ps1 -VideoFolder $env:TRANSCRIBE_VIDEO_FOLDER -SingleVideo "Some Episode.mp4"
```

## Scripts

## `download_video.py`

Downloads from YouTube/VK using `yt-dlp`.

Reads:
- `DOWNLOAD_URL` (or legacy `PLAYLIST_URL`)
- `VIDEO_RESOLUTION`

## `transcribe.py`

Pipeline:
- Whisper transcription (`--language` from `TRANSCRIBE_WHISPER_LANGUAGE`)
- Optional local speech span detection (`video_name.speech_spans.json`)
- Optional sidecar SRT filtering by speech overlaps
- Optional span-aware Whisper transcription with absolute timeline preserved
- MarianMT translation (`TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE` -> `TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE`)
- Optional English phonetic respelling
- UTF-8 + Windows-1251 translated outputs
- Optional sidecar update from translated output (`TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION`)
- Optional move to `TRANSCRIBE_OUTPUT_FOLDER`

## `subtitles_to_markdown.py`

Combines subtitle files from `TRANSCRIBE_SUBTITLE_SOURCE_DIR` into one Markdown file:
- reads files recursively using `TRANSCRIBE_SUBTITLE_GLOB` (default `*.srt`)
- reads source files in `TRANSCRIBE_SUBTITLE_SOURCE_ENCODING`
- retries with `TRANSCRIBE_SUBTITLE_SOURCE_ENCODING_FALLBACKS` when needed
- removes index/timecode lines
- keeps an empty line between subtitle blocks
- writes output to `TRANSCRIBE_SUBTITLE_OUTPUT_MD`
- uses `TRANSCRIBE_SUBTITLE_OUTPUT_ENCODING` (or defaults to sidecar SRT encoding)

> NOTE: Use responsibly and only with content you are allowed to process.

## Requirements

Managed by `uv`. Main dependencies:
- `yt-dlp`
- `transformers`
- `sentencepiece`
- `protobuf`
- `g2p-en`
- `torch`
- `nltk`
- `openai-whisper`
