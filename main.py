"""Master orchestration script for transcription/translation pipeline."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from sidecar_replace import replace_sidecars
from subtitles_to_markdown import merge_subtitles

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STEP-FLAG PROFILES (valid Python constants; only step flags differ)
# Copy selected values into `.env` as TRANSCRIBE_RUN_* flags.
# ---------------------------------------------------------------------------
# Scenario: force-create speech spans only.
# Typical time: ~0.05-0.5 min/video.
PROFILE_FORCE_CREATE_SPANS = {
    'TRANSCRIBE_RUN_GENERATE_SPANS': True,
    'TRANSCRIBE_RUN_FILTER_SIDECARS': False,
    'TRANSCRIBE_RUN_TRANSCRIPTION': False,
    'TRANSCRIBE_RUN_TRANSLATION': False,
    'TRANSCRIBE_RUN_BAKE_SUBTITLES': False,
    'TRANSCRIBE_RUN_SIDECAR_REPLACE': False,
    'TRANSCRIBE_RUN_MERGE': False,
}

# Scenario: create spans and filter sidecars.
# Typical time: ~0.1-0.8 min/video.
PROFILE_FORCE_FILTER_SIDECARS = {
    'TRANSCRIBE_RUN_GENERATE_SPANS': True,
    'TRANSCRIBE_RUN_FILTER_SIDECARS': True,
    'TRANSCRIBE_RUN_TRANSCRIPTION': False,
    'TRANSCRIBE_RUN_TRANSLATION': False,
    'TRANSCRIBE_RUN_BAKE_SUBTITLES': False,
    'TRANSCRIBE_RUN_SIDECAR_REPLACE': False,
    'TRANSCRIBE_RUN_MERGE': False,
}

# Scenario: rerun translation only from sidecar.
# Typical time: ~0.1-1.0 min/video.
PROFILE_FORCE_RERUN_TRANSLATION_FROM_SIDECAR = {
    'TRANSCRIBE_RUN_GENERATE_SPANS': False,
    'TRANSCRIBE_RUN_FILTER_SIDECARS': False,
    'TRANSCRIBE_RUN_TRANSCRIPTION': False,
    'TRANSCRIBE_RUN_TRANSLATION': True,
    'TRANSCRIBE_RUN_BAKE_SUBTITLES': False,
    'TRANSCRIBE_RUN_SIDECAR_REPLACE': False,
    'TRANSCRIBE_RUN_MERGE': False,
}

# Scenario: full rerun (spans + filter + transcribe + translate).
# Typical time: ~1-10+ min/video (Whisper-heavy).
PROFILE_FORCE_RERUN_TRANSCRIBE_AND_TRANSLATE = {
    'TRANSCRIBE_RUN_GENERATE_SPANS': True,
    'TRANSCRIBE_RUN_FILTER_SIDECARS': True,
    'TRANSCRIBE_RUN_TRANSCRIPTION': True,
    'TRANSCRIBE_RUN_TRANSLATION': True,
    'TRANSCRIBE_RUN_BAKE_SUBTITLES': False,
    'TRANSCRIBE_RUN_SIDECAR_REPLACE': False,
    'TRANSCRIBE_RUN_MERGE': False,
}

# Scenario: end-to-end result with bake and merge.
# Typical time: ~2-12+ min/video.
PROFILE_ALL_IN_ONE_WITH_MERGE = {
    'TRANSCRIBE_RUN_GENERATE_SPANS': True,
    'TRANSCRIBE_RUN_FILTER_SIDECARS': True,
    'TRANSCRIBE_RUN_TRANSCRIPTION': True,
    'TRANSCRIBE_RUN_TRANSLATION': True,
    'TRANSCRIBE_RUN_BAKE_SUBTITLES': True,
    'TRANSCRIBE_RUN_SIDECAR_REPLACE': False,
    'TRANSCRIBE_RUN_MERGE': True,
}
# ---------------------------------------------------------------------------

MEMORY_FILE = Path(".master_pipeline_memory.json")


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _variant_for_encoding(encoding: str) -> str:
    normalized = encoding.lower()
    if normalized in {"windows-1251", "cp1251"}:
        return "translated_windows1251"
    return "translated_utf8"


def _target_videos(video_folder: Path, single_video: str | None) -> list[Path]:
    videos = sorted(video_folder.glob("*.mp4"))
    if not single_video:
        return videos

    target = single_video.lower().strip()
    return [
        video
        for video in videos
        if video.name.lower() == target or video.stem.lower() == target
    ]


def _load_memory() -> dict[str, bool]:
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_memory(memory: dict[str, bool]) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=True, indent=2), encoding="utf-8")


def _ask_with_memory(memory: dict[str, bool], key: str, question: str) -> bool:
    if key in memory:
        logger.info(
            '[remembered] %s -> %s',
            question,
            'yes' if memory[key] else 'no',
        )
        return memory[key]

    if not sys.stdin.isatty():
        raise RuntimeError(f'Need user input for "{question}" in non-interactive mode.')

    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            memory[key] = True
            _save_memory(memory)
            return True
        if answer in {"n", "no"}:
            memory[key] = False
            _save_memory(memory)
            return False


def _run_transcribe_step(name: str, env_overrides: dict[str, str]) -> None:
    logger.info('==> Step: %s', name)
    env = os.environ.copy()
    env.update(env_overrides)
    process = subprocess.run(
        [sys.executable, "transcribe.py"],
        env=env,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")


def _run_merge_step(
        video_folder: Path,
        sidecar_encoding: str,
        output_md: Path,
) -> None:
    logger.info('==> Step: merge subtitles')
    variant = _variant_for_encoding(sidecar_encoding)
    report = merge_subtitles(
        source_dir=video_folder,
        output_md=output_md,
        pattern=f"{variant}/*.srt",
        source_encoding=sidecar_encoding,
        output_encoding=sidecar_encoding,
    )
    if report.errors:
        raise RuntimeError("Merge step reported errors.")


def main() -> int:
    video_folder_raw = _first_env("TRANSCRIBE_VIDEO_FOLDER", "VIDEO_FOLDER")
    if not video_folder_raw:
        raise RuntimeError("TRANSCRIBE_VIDEO_FOLDER (or VIDEO_FOLDER) is required.")

    video_folder = Path(video_folder_raw)
    if not video_folder.exists():
        raise RuntimeError(f"Video folder does not exist: {video_folder}")

    single_video = _first_env("TRANSCRIBE_SINGLE_VIDEO", default=None)
    videos = _target_videos(video_folder, single_video)
    if not videos:
        raise RuntimeError("No target videos found.")

    whisper_language = _first_env("TRANSCRIBE_WHISPER_LANGUAGE", "LANGUAGE", default="fr")
    source_lang = _first_env("TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE", default=whisper_language)
    target_lang = _first_env("TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE", default="en")
    translation_model = _first_env("TRANSCRIBE_TRANSLATION_MODEL", default="")
    sidecar_encoding = _first_env(
        "TRANSCRIBE_SIDECAR_SRT_ENCODING",
        "TRANSCRIBE_DUPLICATE_SRT_ENCODING",
        "DUPLICATE_SRT_ENCODING",
        default="utf-8",
    )
    hf_token = _first_env("TRANSCRIBE_HF_TOKEN", "HF_TOKEN", default="")
    offline_mode = _first_env("TRANSCRIBE_OFFLINE_MODE", default="false")
    merge_output_md = Path(
        _first_env(
            "TRANSCRIBE_SUBTITLE_OUTPUT_MD",
            default=str(video_folder / "merged_srt_files.md"),
        )
    )

    run_generate_spans = _to_bool(
        _first_env("TRANSCRIBE_RUN_GENERATE_SPANS", default="false"),
        False,
    )
    run_filter_sidecars = _to_bool(
        _first_env("TRANSCRIBE_RUN_FILTER_SIDECARS", default="false"),
        False,
    )
    run_transcription = _to_bool(
        _first_env("TRANSCRIBE_RUN_TRANSCRIPTION", default="false"),
        False,
    )
    run_translation = _to_bool(
        _first_env("TRANSCRIBE_RUN_TRANSLATION", default="false"),
        False,
    )
    run_bake_subtitles = _to_bool(
        _first_env("TRANSCRIBE_RUN_BAKE_SUBTITLES", default="false"),
        False,
    )
    run_sidecar_replace = _to_bool(
        _first_env("TRANSCRIBE_RUN_SIDECAR_REPLACE", default="false"),
        False,
    )
    run_merge = _to_bool(_first_env("TRANSCRIBE_RUN_MERGE", default="false"), False)

    force_spans = _to_bool(_first_env("TRANSCRIBE_FORCE_SPANS", default="true"), True)
    force_transcription = _to_bool(_first_env("TRANSCRIBE_FORCE_TRANSCRIPTION", default="false"), False)
    force_translation = _to_bool(_first_env("TRANSCRIBE_FORCE_TRANSLATION", default="true"), True)
    force_bake_subtitles = _to_bool(
        _first_env("TRANSCRIBE_FORCE_BAKE_SUBTITLES", default="false"),
        False,
    )
    append_source = _to_bool(_first_env("TRANSCRIBE_TRANSLATION_APPEND_SOURCE", default="true"), True)

    logger.info(
        'Effective step flags: generate_spans=%s, filter_sidecars=%s, '
        'transcription=%s, translation=%s, bake_subtitles=%s, '
        'sidecar_replace=%s, merge=%s',
        run_generate_spans,
        run_filter_sidecars,
        run_transcription,
        run_translation,
        run_bake_subtitles,
        run_sidecar_replace,
        run_merge,
    )

    if not any(
        [
            run_generate_spans,
            run_filter_sidecars,
            run_transcription,
            run_translation,
            run_bake_subtitles,
            run_sidecar_replace,
            run_merge,
        ]
    ):
        logger.warning('No steps enabled. Set TRANSCRIBE_RUN_* flags in .env.')
        return 0

    memory = _load_memory()

    common_env = {
        "TRANSCRIBE_VIDEO_FOLDER": str(video_folder),
        "TRANSCRIBE_SINGLE_VIDEO": single_video or "",
        "TRANSCRIBE_WHISPER_LANGUAGE": whisper_language or "",
        "TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE": source_lang or "",
        "TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE": target_lang or "",
        "TRANSCRIBE_TRANSLATION_MODEL": translation_model or "",
        "TRANSCRIBE_SIDECAR_SRT_ENCODING": sidecar_encoding or "utf-8",
        "TRANSCRIBE_HF_TOKEN": hf_token or "",
        "TRANSCRIBE_OFFLINE_MODE": offline_mode or "false",
        "TRANSCRIBE_OUTPUT_FOLDER": "",
        "TRANSCRIBE_TRANSLATION_APPEND_SOURCE": "true" if append_source else "false",
        "TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION": "false",
    }

    if run_filter_sidecars:
        missing_spans = [
            video for video in videos if not video.with_suffix(".speech_spans.json").exists()
        ]
        if missing_spans and not run_generate_spans:
            if _ask_with_memory(
                    memory,
                    "missing_spans_for_filter",
                    "Missing speech spans for filter step. Run spans generation first?",
            ):
                run_generate_spans = True
            else:
                run_filter_sidecars = False

    if run_sidecar_replace:
        variant = _variant_for_encoding(sidecar_encoding or "utf-8")
        missing_variant = [
            video
            for video in videos
            if not (video.parent / video.stem / variant / f"{video.stem}.srt").exists()
        ]
        if missing_variant and not run_translation:
            if _ask_with_memory(
                    memory,
                    "missing_variant_for_replace",
                    f'Missing "{variant}" subtitles for sidecar replace. Run translation first?',
            ):
                run_translation = True
            else:
                run_sidecar_replace = False

    if run_bake_subtitles:
        variant = _variant_for_encoding(sidecar_encoding or "utf-8")
        missing_variant = [
            video
            for video in videos
            if not (video.parent / video.stem / variant / f"{video.stem}.srt").exists()
        ]
        if missing_variant and not run_translation:
            if _ask_with_memory(
                    memory,
                    "missing_variant_for_bake",
                    f'Missing "{variant}" subtitles for bake step. Run translation first?',
            ):
                run_translation = True
            else:
                run_bake_subtitles = False

    if run_generate_spans:
        _run_transcribe_step(
            "generate spans",
            {
                **common_env,
                "TRANSCRIBE_ENABLE_SPEECH_SPANS": "true",
                "TRANSCRIBE_SPEECH_SPANS_ONLY_MODE": "true",
                "TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS": "false",
                "TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING": "true",
                "TRANSCRIBE_SPEECH_SPANS_OVERWRITE": "true" if force_spans else "false",
            },
        )

    if run_filter_sidecars:
        _run_transcribe_step(
            "filter sidecars",
            {
                **common_env,
                "TRANSCRIBE_ENABLE_SPEECH_SPANS": "true",
                "TRANSCRIBE_SPEECH_SPANS_ONLY_MODE": "true",
                "TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS": "true",
                "TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING": "true",
                "TRANSCRIBE_SPEECH_SPANS_OVERWRITE": "true" if force_spans else "false",
            },
        )

    if run_transcription:
        if force_transcription:
            logger.info('==> Step: force transcription cleanup')
            for video in videos:
                output_dir = video.parent / video.stem
                if output_dir.exists():
                    for child in output_dir.iterdir():
                        if child.name in {"original", "stdout.txt"}:
                            if child.is_dir():
                                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(child)], check=False)
                            else:
                                child.unlink(missing_ok=True)

        _run_transcribe_step(
            "transcription",
            {
                **common_env,
                "TRANSCRIBE_ENABLE_SPEECH_SPANS": "true",
                "TRANSCRIBE_SPEECH_SPANS_ONLY_MODE": "false",
                "TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS": "true",
                "TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER": "true",
                "TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING": "true",
                "TRANSCRIBE_SPEECH_SPANS_OVERWRITE": "true" if force_spans else "false",
                "TRANSCRIBE_TRANSLATION_ONLY_MODE": "false",
                "TRANSCRIBE_ENABLE_TRANSLATION": "false",
            },
        )

    if run_translation:
        missing_sidecars = [video for video in videos if not video.with_suffix(".srt").exists()]
        if missing_sidecars:
            if not _ask_with_memory(
                    memory,
                    "missing_sidecars_for_translation",
                    "Some sidecar SRT files are missing. Continue translation step anyway?",
            ):
                run_translation = False

    if run_translation:
        _run_transcribe_step(
            "translation-only from sidecar",
            {
                **common_env,
                "TRANSCRIBE_ENABLE_SPEECH_SPANS": "false",
                "TRANSCRIBE_SPEECH_SPANS_ONLY_MODE": "false",
                "TRANSCRIBE_TRANSLATION_ONLY_MODE": "true",
                "TRANSCRIBE_TRANSLATION_INPUT": "sidecar",
                "TRANSCRIBE_TRANSLATION_OVERWRITE": "true" if force_translation else "false",
            },
        )

    if run_bake_subtitles:
        _run_transcribe_step(
            "bake target subtitles into video",
            {
                **common_env,
                "TRANSCRIBE_ENABLE_SPEECH_SPANS": "false",
                "TRANSCRIBE_SPEECH_SPANS_ONLY_MODE": "false",
                "TRANSCRIBE_TRANSLATION_ONLY_MODE": "false",
                "TRANSCRIBE_ENABLE_TRANSLATION": "false",
                "TRANSCRIBE_ENABLE_BAKED_SUBTITLES": "true",
                "TRANSCRIBE_BAKE_SUBTITLES_ONLY_MODE": "true",
                "TRANSCRIBE_BAKE_SUBTITLES_OVERWRITE": "true" if force_bake_subtitles else "false",
            },
        )

    if run_merge:
        # Merge result should reflect processing outputs, not sidecar replace result.
        _run_merge_step(
            video_folder=video_folder, sidecar_encoding=sidecar_encoding or "utf-8", output_md=merge_output_md)

    if run_sidecar_replace:
        logger.info('==> Step: sidecar replace')
        replace_sidecars(
            video_folder=video_folder,
            variant=_variant_for_encoding(sidecar_encoding or "utf-8"),
            single_video=single_video,
        )

    logger.info('Master pipeline finished.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
