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

2. Create a `.env` file (copy from `.env.example` if available) and configure:
```env
PLAYLIST_URL=<your-youtube-playlist-url>
VIDEO_RESOLUTION=720p
LANGUAGE=en
INPUT_SRT_FOLDER=path/to/input/srt/folder
OUTPUT_SRT_FOLDER=path/to/output/srt/folder
```

3. Run the downloader:
```bash
python download_video.py
```
> NOTE: I do not encourage to download any copyright content from Youtube, this script is for educational purpose only.

# Scripts

## download_video.py - YouTube Playlist Downloader

Downloads entire YouTube playlists using pytube's native `Playlist` class.

**Configuration** (in `.env`):
- `PLAYLIST_URL` - YouTube playlist URL
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


# Requirements
Managed by `uv`. Dependencies include:
- pytube
- requests
- transformers (for translation)
- g2p-en (for phonetic transcription)
- torch


# How it works?

## download_video.py
Uses pytube's native `Playlist` class to fetch all videos from a playlist and download them sequentially.

## translate_srt.py
- Loads MarianMT model (`Helsinki-NLP/opus-mt-en-ru`) for English-to-Russian translation
- Uses `g2p-en` for grapheme-to-phoneme conversion
- Processes subtitles in batches for efficiency
- Outputs multiline subtitles: original + pronunciation + translation
