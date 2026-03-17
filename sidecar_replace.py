"""Replace sidecar SRT files from a selected per-video subtitle variant."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def replace_sidecars(video_folder: Path, variant: str, single_video: str | None = None) -> None:
    videos = list(video_folder.glob("*.mp4"))
    if single_video:
        target = single_video.lower().strip()
        videos = [
            video
            for video in videos
            if video.name.lower() == target or video.stem.lower() == target
        ]

    if not videos:
        logger.warning("No .mp4 files found in %s", video_folder)
        return

    logger.info("Found %d video(s) for sidecar replace", len(videos))
    replaced = 0

    for video in videos:
        stem = video.stem
        source = video_folder / stem / variant / f"{stem}.srt"
        target = video_folder / f"{stem}.srt"

        if not source.exists():
            logger.warning("Missing source variant SRT: %s", source)
            continue

        shutil.copyfile(source, target)
        replaced += 1
        logger.info("Replaced sidecar: %s <- %s", target.name, source)

    logger.info("Done. Replaced %d/%d sidecar file(s).", replaced, len(videos))


def main() -> None:
    video_folder_raw = _first_env("TRANSCRIBE_VIDEO_FOLDER", "VIDEO_FOLDER")
    if not video_folder_raw:
        raise RuntimeError(
            "TRANSCRIBE_VIDEO_FOLDER (or legacy VIDEO_FOLDER) is not set in .env"
        )

    variant = _first_env("TRANSCRIBE_SIDECAR_REPLACE_VARIANT", default="translated_utf8")
    if not variant:
        raise RuntimeError("TRANSCRIBE_SIDECAR_REPLACE_VARIANT is not set")

    single_video = _first_env("TRANSCRIBE_SINGLE_VIDEO", default=None)
    replace_sidecars(
        video_folder=Path(video_folder_raw),
        variant=variant,
        single_video=single_video,
    )


if __name__ == "__main__":
    main()
