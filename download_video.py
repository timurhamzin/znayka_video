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


FIX_OPUS_AUDIO = _env_flag('DOWNLOAD_FIX_OPUS_AUDIO', default=False)


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


def _detect_audio_codec(file_path: Path) -> str | None:
    result = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'a:0',
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


def _replace_opus_audio_with_aac(file_path: Path) -> bool:
    temp_path = file_path.with_name(f'{file_path.stem}.aacfix{file_path.suffix}')
    result = subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(file_path),
            '-map',
            '0',
            '-c:v',
            'copy',
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
        logger.error('Failed to convert audio to AAC for %s', file_path)
        return False

    temp_path.replace(file_path)
    return True


def _fix_opus_audio(files: set[Path]) -> None:
    if not files:
        logger.info('No downloaded files were detected for Opus check.')
        return

    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
        logger.warning(
            'DOWNLOAD_FIX_OPUS_AUDIO=true, but ffprobe/ffmpeg are not available. '
            'Skipping Opus fix.'
        )
        return

    for file_path in sorted(files):
        if not file_path.exists():
            continue
        codec = _detect_audio_codec(file_path)
        if codec != 'opus':
            continue
        logger.info('Detected Opus audio in %s. Re-encoding audio to AAC.', file_path)
        _replace_opus_audio_with_aac(file_path)


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

    if FIX_OPUS_AUDIO:
        files_to_check = _collect_output_files(info)
        files_to_check.update(downloaded_files)
        _fix_opus_audio(files_to_check)


if __name__ == '__main__':

    if not DOWNLOAD_URL:
        raise RuntimeError('DOWNLOAD_URL (or PLAYLIST_URL) not set in .env')

    download_from_url(DOWNLOAD_URL, USER_RES)
