import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yt_dlp
from dotenv import load_dotenv
from yt_dlp.utils import DownloadError


load_dotenv()

BASE_DIR = Path(__file__).parent

DOWNLOAD_URL = os.environ.get('DOWNLOAD_URL') or os.environ.get('PLAYLIST_URL')
USER_RES = os.environ.get('VIDEO_RESOLUTION', '720p')


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


FIX_INCOMPATIBLE_MEDIA = _env_flag('DOWNLOAD_FIX_OPUS_AUDIO', default=False)


def _normalize_resolution(resolution: str) -> str:
    value = resolution.lower().strip()
    if value.endswith('p'):
        value = value[:-1]

    if not value.isdigit():
        raise ValueError('VIDEO_RESOLUTION must be a number like 360 or 720')

    return value


def _collect_output_files(info: dict[str, Any] | None) -> set[Path]:
    if not info:
        return set()

    paths: set[Path] = set()

    entries = info.get('entries')
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                paths.update(_collect_output_files(entry))

    requested_downloads = info.get('requested_downloads')
    if isinstance(requested_downloads, list):
        for item in requested_downloads:
            if not isinstance(item, dict):
                continue
            filepath = item.get('filepath')
            if isinstance(filepath, str):
                paths.add(Path(filepath))

    for key in ('_filename', 'filepath'):
        candidate = info.get(key)
        if isinstance(candidate, str):
            paths.add(Path(candidate))

    return paths


def _detect_stream_codec(file_path: Path, stream_selector: str) -> str | None:
    result = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            stream_selector,
            '-show_entries',
            'stream=codec_name',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            str(file_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    codec = result.stdout.strip().lower()
    return codec or None


def _convert_to_browser_compatible_mp4(file_path: Path) -> Path | None:
    output_path = file_path.with_suffix('.mp4')
    temp_path = output_path.with_name(f'{output_path.stem}.compatfix.mp4')

    result = subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(file_path),
            '-map',
            '0:v:0',
            '-map',
            '0:a:0?',
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-crf',
            '22',
            '-pix_fmt',
            'yuv420p',
            '-c:a',
            'aac',
            '-b:a',
            '192k',
            '-movflags',
            '+faststart',
            str(temp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(
            'Failed to convert media to browser-compatible MP4 for %s',
            file_path,
        )
        return None

    if output_path.exists():
        output_path.unlink()
    temp_path.replace(output_path)

    if output_path != file_path and file_path.exists():
        file_path.unlink()

    return output_path


def _fix_incompatible_media(files: set[Path]) -> None:
    if not files:
        logger.info('No downloaded files were detected for media compatibility check.')
        return

    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
        logger.warning(
            'DOWNLOAD_FIX_OPUS_AUDIO=true, but ffprobe/ffmpeg are not available. '
            'Skipping media compatibility fix.'
        )
        return

    for file_path in sorted(files):
        if not file_path.exists():
            continue

        video_codec = _detect_stream_codec(file_path, 'v:0')
        audio_codec = _detect_stream_codec(file_path, 'a:0')

        needs_fix = audio_codec == 'opus' or video_codec != 'h264'
        if not needs_fix:
            continue

        logger.info(
            'Detected potentially incompatible streams in %s (video=%s, audio=%s). '
            'Converting to H.264/AAC MP4.',
            file_path,
            video_codec or 'unknown',
            audio_codec or 'none',
        )
        output_path = _convert_to_browser_compatible_mp4(file_path)
        if output_path is not None:
            logger.info('Compatibility-fixed file: %s', output_path)


def download_from_url(url: str, resolution: str) -> None:

    save_path = BASE_DIR / 'downloads'
    save_path.mkdir(exist_ok=True)

    height = _normalize_resolution(resolution)
    downloaded_files: set[Path] = set()

    def _progress_hook(status: dict[str, Any]) -> None:
        if status.get('status') != 'finished':
            return
        filename = status.get('filename')
        if isinstance(filename, str):
            downloaded_files.add(Path(filename))

    ydl_opts = {
        'format': f'bestvideo[height<={height}]+bestaudio/best[height<={height}]',
        'outtmpl': str(
            save_path / '%(extractor_key|source)s/%(playlist|single)s/%(title)s.%(ext)s'
        ),
        'merge_output_format': 'mp4',
        'ignoreerrors': True,
        'js_runtimes': {
            'node': {
                'path': r'C:\Program Files\nodejs\node.exe'
            }
        },
        'remote_components': 'ejs:github',
        'progress_hooks': [_progress_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as error:
        logger.error('Failed to download URL: %s', url)
        raise RuntimeError('Download failed. Check URL and access permissions.') from error

    if FIX_INCOMPATIBLE_MEDIA:
        files_to_check = _collect_output_files(info)
        files_to_check.update(downloaded_files)
        _fix_incompatible_media(files_to_check)


if __name__ == '__main__':

    if not DOWNLOAD_URL:
        raise RuntimeError('DOWNLOAD_URL (or PLAYLIST_URL) not set in .env')

    download_from_url(DOWNLOAD_URL, USER_RES)
