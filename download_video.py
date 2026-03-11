import logging
import os
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).parent

PLAYLIST_URL = os.environ.get('PLAYLIST_URL')
USER_RES = os.environ.get('VIDEO_RESOLUTION', '720p')


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def download_playlist(url: str, resolution: str) -> None:

    save_path = BASE_DIR / 'downloads'
    save_path.mkdir(exist_ok=True)

    height = resolution.replace('p', '')

    ydl_opts = {
        'format': f'bestvideo[height<={height}]+bestaudio/best[height<={height}]',
        'outtmpl': str(save_path / '%(playlist)s/%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'ignoreerrors': True,
        'js_runtimes': {
            'node': {
                'path': r'C:\Program Files\nodejs\node.exe'
            }
        },
        'remote_components': 'ejs:github',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == '__main__':

    if not PLAYLIST_URL:
        raise RuntimeError('PLAYLIST_URL not set in .env')

    download_playlist(PLAYLIST_URL, USER_RES)