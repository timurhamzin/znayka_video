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
.\run_download.ps1
.\run_transcribe.ps1
.\run_subtitles_to_markdown.ps1
```

### Linux/macOS (bash)

```bash
chmod +x run_download.sh run_transcribe.sh run_subtitles_to_markdown.sh
./run_download.sh
./run_transcribe.sh
./run_subtitles_to_markdown.sh
```

## French Video -> English Translation Example

Set:

```env
TRANSCRIBE_WHISPER_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE=fr
TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE=en
```

This will transcribe French audio and translate subtitles to English.

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
- Sidecar translated SRT in `TRANSCRIBE_SIDECAR_SRT_ENCODING`
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
