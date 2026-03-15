# Setup

## Windows Prerequisites

Install Node.js runtime (required for YouTube extraction):

```powershell
winget install OpenJS.NodeJS
```

This resolves the warning: `WARNING: [youtube] No supported JavaScript runtime could be found`.

## Install Dependencies

```bash
uv sync
```

## Download NLTK Data (Required for phonetic transcription)

Run once after installation:

```bash
python -m nltk.downloader averaged_perceptron_tagger cmudict punkt
```

Or download programmatically:

```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
nltk.download('punkt')
```

## Configure Environment

1. Create a `.env` file (copy from `.env.example` if available) and configure:

```env
# Video processing
VIDEO_FOLDER=path/to/your/videos
DUPLICATE_SRT_ENCODING=utf-8
LANGUAGE=en

# HuggingFace token (optional, for faster model downloads)
HF_TOKEN=your_huggingface_token

# Output folder for processed videos (optional)
OUTPUT_FOLDER=path/to/output/folder
```

2. Run the transcriber:

```bash
python transcribe.py
```

> NOTE: I do not encourage to download any copyright content from Youtube, this script is for educational purpose only.

# Scripts

## download_video.py - Video Downloader (YouTube/VK)

Downloads videos with `yt-dlp` from a URL. Supports both playlist and single-video links.

**Configuration** (in `.env`):
- `DOWNLOAD_URL` - YouTube or VK URL (playlist or single video)
- `PLAYLIST_URL` - Legacy fallback variable (optional)
- `VIDEO_RESOLUTION` - Video quality (`360p` or `720p`)

**Usage**:
```bash
python download_video.py
```

## translate_srt.py - SRT Subtitle Translator

Translates SRT subtitle files from English to Russian using MarianMT model.
Optionally adds phonetic transcription (respelling) for pronunciation help.

**Features**:
- Batch translation for efficiency
- Phonetic respelling with stress markers (for English)
- Append mode: keeps original + adds pronunciation + translation

**Configuration** (in `.env`):
- `LANGUAGE` - Phonetics mode: `en` for detailed English respelling with syllables/stress
- `INPUT_SRT_FOLDER` - Folder containing input `.srt` files
- `OUTPUT_SRT_FOLDER` - Folder for translated `.srt` files (created if doesn't exist)

**Usage**:
```bash
# With English phonetic respelling (syllables + stress markers)
LANGUAGE=en python translate_srt.py

# With simple phonetic transcription
LANGUAGE=other python translate_srt.py
```

The script processes all `.srt` files in the input folder and saves translated versions to the output folder.

## transcribe.py - Video Transcription and Translation Pipeline

Complete pipeline for transcribing videos with Whisper and translating subtitles.

**Features**:
- Whisper CLI transcription with real-time progress output
- English-to-Russian translation with MarianMT
- Phonetic respelling with syllables and stress markers
- Multi-encoding output (UTF-8 and Windows-1251)
- Skips already processed steps (resume support)
- Optional: move completed videos to output folder

**Configuration** (in `.env`):
- `VIDEO_FOLDER` - Folder containing `.mp4` video files
- `LANGUAGE` - Transcription language: `en` for English
- `DUPLICATE_SRT_ENCODING` - Encoding for duplicate SRT file (e.g., `utf-8`, `windows-1251`)
- `HF_TOKEN` - HuggingFace token (optional, for faster model downloads)
- `OUTPUT_FOLDER` - Destination folder for processed videos (optional)

**Directory structure created**:
```
video_folder/
├── video_file.mp4
├── video_file.srt (translated, in DUPLICATE_SRT_ENCODING)
└── video_file_name/
    ├── original/
    │   └── video_file.srt (original transcription)
    ├── stdout.txt (Whisper output)
    ├── translated_windows1251/
    │   └── video_file.srt
    └── translated_utf8/
        └── video_file.srt
```

**Usage**:
```bash
python transcribe.py
```

The script processes videos one by one, showing progress in the console.


# Requirements
Managed by `uv`. Dependencies include:
- pytube
- requests
- transformers (for translation)
- g2p-en (for phonetic transcription)
- torch
- nltk (for text processing)
- openai-whisper (for transcription)


# How it works?

## download_video.py
Uses `yt-dlp` to download media from a provided URL (playlist or single item).

## translate_srt.py
- Loads MarianMT model (`Helsinki-NLP/opus-mt-en-ru`) for English-to-Russian translation
- Uses `g2p-en` for grapheme-to-phoneme conversion
- Processes subtitles in batches for efficiency
- Outputs multiline subtitles: original + pronunciation + translation

## transcribe.py
- **WhisperTranscriber**: Runs Whisper CLI, streams output to console
- **TranslationModel**: Loads MarianMT for batch translation
- **EnglishRespeller**: Converts phonemes to readable respelling with stress
- **SRTTranslator**: Parses SRT, translates text lines in batches
- **VideoPipeline**: Orchestrates the workflow, handles caching and file moves
