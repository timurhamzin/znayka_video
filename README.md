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
TRANSCRIBE_DUPLICATE_SRT_ENCODING=utf-8
TRANSCRIBE_HF_TOKEN=your_hf_token
TRANSCRIBE_OUTPUT_FOLDER=
```

Notes:
- If `TRANSCRIBE_TRANSLATION_MODEL` is empty, model is auto-selected from source/target language pair.
- English phonetic respelling works only when source language is English. For French source, phonetic lines are skipped.
- Legacy vars (`VIDEO_FOLDER`, `LANGUAGE`, `DUPLICATE_SRT_ENCODING`, `HF_TOKEN`, `OUTPUT_FOLDER`) are still supported for backward compatibility.

## Run Scripts (activate `.venv` automatically)

### Windows (PowerShell)

```powershell
./run_download.ps1
./run_transcribe.ps1
```

### Linux/macOS (bash)

```bash
chmod +x run_download.sh run_transcribe.sh
./run_download.sh
./run_transcribe.sh
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
- MarianMT translation (`TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE` -> `TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE`)
- Optional English phonetic respelling
- UTF-8 + Windows-1251 translated outputs
- Duplicate SRT in `TRANSCRIBE_DUPLICATE_SRT_ENCODING`
- Optional move to `TRANSCRIBE_OUTPUT_FOLDER`

> NOTE: Use responsibly and only with content you are allowed to process.

## Requirements

Managed by `uv`. Main dependencies:
- `yt-dlp`
- `transformers`
- `g2p-en`
- `torch`
- `nltk`
- `openai-whisper`
