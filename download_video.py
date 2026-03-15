import logging
import os
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from yt_dlp.utils import DownloadError


load_dotenv()

BASE_DIR = Path(__file__).parent

DOWNLOAD_URL = os.environ.get('DOWNLOAD_URL') or os.environ.get('PLAYLIST_URL')
USER_RES = os.environ.get('VIDEO_RESOLUTION', '720p')


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _normalize_resolution(resolution: str) -> str:
    value = resolution.lower().strip()
    if value.endswith('p'):
        value = value[:-1]

    if not value.isdigit():
        raise ValueError('VIDEO_RESOLUTION must be a number like 360 or 720')

    return value


def download_from_url(url: str, resolution: str) -> None:

    save_path = BASE_DIR / 'downloads'
    save_path.mkdir(exist_ok=True)

    height = _normalize_resolution(resolution)

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
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except DownloadError as error:
        logger.error('Failed to download URL: %s', url)
        raise RuntimeError('Download failed. Check URL and access permissions.') from error


if __name__ == '__main__':

    if not DOWNLOAD_URL:
        raise RuntimeError('DOWNLOAD_URL (or PLAYLIST_URL) not set in .env')

    download_from_url(DOWNLOAD_URL, USER_RES)
