"""Master orchestration script for transcription/translation pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
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

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

FEATURE_ORDER = [
    ('TRANSCRIBE_RUN_GENERATE_SPANS', 'Generate speech spans'),
    ('TRANSCRIBE_RUN_FILTER_SIDECARS', 'Filter sidecar by spans'),
    ('TRANSCRIBE_RUN_TRANSCRIPTION', 'Run Whisper transcription'),
    ('TRANSCRIBE_RUN_TRANSLATION', 'Run translation from sidecar'),
    ('TRANSCRIBE_RUN_MERGE', 'Merge subtitles to Markdown'),
    ('TRANSCRIBE_RUN_SIDECAR_REPLACE', 'Replace sidecar from variant'),
    ('TRANSCRIBE_RUN_BAKE_SUBTITLES', 'Bake subtitles into video'),
]
FEATURE_DETAILS = {
    'TRANSCRIBE_RUN_GENERATE_SPANS': (
        'Create/update speech span JSON files only.',
        '~0.05-0.5 min/video',
    ),
    'TRANSCRIBE_RUN_FILTER_SIDECARS': (
        'Filter sidecar SRT by speech spans.',
        '~0.05-0.3 min/video',
    ),
    'TRANSCRIBE_RUN_TRANSCRIPTION': (
        'Whisper transcription pass.',
        '~1-8+ min/video',
    ),
    'TRANSCRIBE_RUN_TRANSLATION': (
        'Translation from sidecar subtitles.',
        '~0.1-1.0 min/video',
    ),
    'TRANSCRIBE_RUN_MERGE': (
        'Merge translated subtitles into markdown report.',
        '~0.01-0.1 min/video',
    ),
    'TRANSCRIBE_RUN_SIDECAR_REPLACE': (
        'Copy selected variant back to sidecar SRT.',
        '~0.01-0.05 min/video',
    ),
    'TRANSCRIBE_RUN_BAKE_SUBTITLES': (
        'Burn target subtitles into video output.',
        '~0.5-3+ min/video',
    ),
}

MEMORY_FILE = Path('.master_pipeline_memory.json')
VIDEO_HEADER_RE = re.compile(r'Video\s+(\d+)/(\d+):\s*(.+)$')


class _Keys:
    UP = 'up'
    DOWN = 'down'
    ENTER = 'enter'
    SPACE = 'space'
    QUIT = 'quit'


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != '':
            return value.strip()
    return default


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'on'}


def _variant_for_encoding(encoding: str) -> str:
    normalized = encoding.lower()
    if normalized in {'windows-1251', 'cp1251'}:
        return 'translated_windows1251'
    return 'translated_utf8'


def _target_videos(video_folder: Path, single_video: str | None) -> list[Path]:
    videos = sorted(video_folder.glob('*.mp4'))
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
        return json.loads(MEMORY_FILE.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _save_memory(memory: dict[str, bool]) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=True, indent=2), encoding='utf-8')


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
        answer = input(f'{question} [y/n]: ').strip().lower()
        if answer in {'y', 'yes'}:
            memory[key] = True
            _save_memory(memory)
            return True
        if answer in {'n', 'no'}:
            memory[key] = False
            _save_memory(memory)
            return False


def _clear_screen() -> None:
    sys.stdout.write('\x1b[2J\x1b[H')
    sys.stdout.flush()


def _enable_windows_vt_mode() -> None:
    if os.name != 'nt':
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        return


def _read_key() -> str:
    if os.name == 'nt':
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in {'\r', '\n'}:
                return _Keys.ENTER
            if ch == ' ':
                return _Keys.SPACE
            if ch.lower() == 'q':
                return _Keys.QUIT
            if ch in {'\x00', '\xe0'}:
                special = msvcrt.getwch()
                if special == 'H':
                    return _Keys.UP
                if special == 'P':
                    return _Keys.DOWN
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch in {'\r', '\n'}:
                return _Keys.ENTER
            if ch == ' ':
                return _Keys.SPACE
            if ch.lower() == 'q':
                return _Keys.QUIT
            if ch == '\x1b':
                seq = sys.stdin.read(2)
                if seq == '[A':
                    return _Keys.UP
                if seq == '[B':
                    return _Keys.DOWN
                return _Keys.QUIT
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return ''


def _select_steps_interactive(initial_flags: dict[str, bool]) -> dict[str, bool] | None:
    _enable_windows_vt_mode()
    index = 0
    flags = dict(initial_flags)

    while True:
        _clear_screen()
        print('Select steps (UP/DOWN, SPACE toggle, ENTER run, q exit):')
        print('')
        for i, (env_key, label) in enumerate(FEATURE_ORDER):
            marker = '>' if i == index else ' '
            checked = 'x' if flags.get(env_key, False) else ' '
            print(f'{marker} [{checked}] {label}')

        selected_key, selected_label = FEATURE_ORDER[index]
        selected_description, selected_eta = FEATURE_DETAILS[selected_key]
        print('')
        print(f'Step: {selected_label}')
        print(f'Description: {selected_description}')
        print(f'Approximate time: {selected_eta}')

        enabled_steps = [
            label for env_key, label in FEATURE_ORDER if flags.get(env_key, False)
        ]
        if enabled_steps:
            print('Run order:')
            for step_num, step in enumerate(enabled_steps, start=1):
                print(f'  {step_num}. {step}')
        else:
            print('Run order: no steps selected')

        key = _read_key()
        if key == _Keys.UP:
            index = (index - 1) % len(FEATURE_ORDER)
        elif key == _Keys.DOWN:
            index = (index + 1) % len(FEATURE_ORDER)
        elif key == _Keys.SPACE:
            flags[selected_key] = not flags.get(selected_key, False)
        elif key == _Keys.ENTER:
            _clear_screen()
            return flags
        elif key == _Keys.QUIT:
            _clear_screen()
            return None


def _edit_features_text(initial_flags: dict[str, bool]) -> dict[str, bool] | None:
    flags = dict(initial_flags)
    print('')
    print('Toggle features in run order (y/n, Enter keeps current, q exits):')
    for env_key, label in FEATURE_ORDER:
        current = flags.get(env_key, False)
        default_hint = 'Y' if current else 'N'
        description, eta = FEATURE_DETAILS[env_key]
        print(f'- {label}: {description} ({eta})')
        while True:
            answer = input(f'{label}? [{default_hint}] ').strip().lower()
            if answer == 'q':
                return None
            if answer == '':
                break
            if answer in {'y', 'yes'}:
                flags[env_key] = True
                break
            if answer in {'n', 'no'}:
                flags[env_key] = False
                break
            print('Please answer y, n, q, or Enter.')
    return flags


def _resolve_interactive_flags(env_flags: dict[str, bool]) -> dict[str, bool] | None:
    interactive = _to_bool(_first_env('TRANSCRIBE_INTERACTIVE', default='true'), True)
    if not interactive:
        return env_flags

    try:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return _select_steps_interactive(env_flags)

        logger.info('Switching to text prompts for step selection.')
        return _edit_features_text(env_flags)
    except (EOFError, KeyboardInterrupt):
        logger.warning('Interactive selection canceled. Falling back to .env flags.')
        return env_flags

    logger.warning('Interactive selection failed. Falling back to .env flags.')
    return env_flags


def _run_transcribe_step(
    name: str,
    env_overrides: dict[str, str],
    step_index: int,
    total_steps: int,
) -> None:
    env = os.environ.copy()
    env.update(env_overrides)
    env['PYTHONUNBUFFERED'] = '1'

    if not HAS_RICH or not sys.stdout.isatty():
        logger.info('==> Step %d/%d: %s', step_index, total_steps, name)
        process = subprocess.run([sys.executable, 'transcribe.py'], env=env, check=False)
        if process.returncode != 0:
            raise RuntimeError(f'Step failed: {name}')
        return

    console = Console()
    logger.info('==> Step %d/%d: %s', step_index, total_steps, name)

    process = subprocess.Popen(
        [sys.executable, 'transcribe.py'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start_time = time.perf_counter()
    current_video = 0
    total_videos = 0
    current_video_name = 'waiting for first video...'

    with Progress(
        SpinnerColumn(),
        TextColumn('[bold cyan]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('{task.fields[stats]}'),
        console=console,
        transient=True,
    ) as progress:
        feature_task = progress.add_task(
            f'Feature {step_index}/{total_steps}: {name}',
            total=1,
            completed=0,
            stats='videos left: n/a',
        )
        video_task = progress.add_task(
            'Video: waiting',
            total=1,
            completed=0,
            stats='elapsed: 0s | eta: n/a | videos left: n/a',
        )

        if process.stdout is None:
            raise RuntimeError('Failed to capture step output.')

        for raw_line in process.stdout:
            line = raw_line.rstrip('\n')
            match = VIDEO_HEADER_RE.search(line)
            if match:
                current_video = int(match.group(1))
                total_videos = int(match.group(2))
                current_video_name = match.group(3).strip()

                completed_videos = max(current_video - 1, 0)
                videos_left = max(total_videos - completed_videos, 0)
                elapsed = time.perf_counter() - start_time
                eta = None
                if completed_videos > 0:
                    per_video = elapsed / completed_videos
                    eta = per_video * videos_left

                eta_text = f'{eta/60:.1f}m' if eta is not None else 'n/a'
                stats_text = (
                    f'elapsed: {elapsed/60:.1f}m | eta: {eta_text} | '
                    f'videos left: {videos_left}'
                )

                progress.update(
                    feature_task,
                    total=total_videos,
                    completed=completed_videos,
                    stats=f'videos left: {videos_left}',
                )
                progress.update(
                    video_task,
                    description=f'Video {current_video}/{total_videos}: {current_video_name}',
                    total=total_videos,
                    completed=completed_videos,
                    stats=stats_text,
                )
                continue

            if any(token in line for token in ('ERROR', 'Traceback', 'Failed')):
                console.print(f'[red]{line}[/red]')

        return_code = process.wait()
        if total_videos > 0:
            progress.update(feature_task, completed=total_videos)
            progress.update(video_task, completed=total_videos)
        else:
            progress.update(feature_task, completed=1)
            progress.update(video_task, description='Video: completed', completed=1)

    if return_code != 0:
        raise RuntimeError(f'Step failed: {name}')


def _run_merge_step(
    video_folder: Path,
    sidecar_encoding: str,
    output_md: Path,
    step_index: int,
    total_steps: int,
) -> None:
    logger.info('==> Step %d/%d: merge subtitles', step_index, total_steps)
    start = time.perf_counter()
    variant = _variant_for_encoding(sidecar_encoding)
    report = merge_subtitles(
        source_dir=video_folder,
        output_md=output_md,
        pattern=f'{variant}/*.srt',
        source_encoding=sidecar_encoding,
        output_encoding=sidecar_encoding,
    )
    elapsed = time.perf_counter() - start
    logger.info('Merge finished in %.1fs', elapsed)
    if report.errors:
        raise RuntimeError('Merge step reported errors.')


def _run_sidecar_replace_step(
    video_folder: Path,
    sidecar_encoding: str,
    single_video: str | None,
    step_index: int,
    total_steps: int,
) -> None:
    logger.info('==> Step %d/%d: sidecar replace', step_index, total_steps)
    start = time.perf_counter()
    replace_sidecars(
        video_folder=video_folder,
        variant=_variant_for_encoding(sidecar_encoding),
        single_video=single_video,
    )
    elapsed = time.perf_counter() - start
    logger.info('Sidecar replace finished in %.1fs', elapsed)


def main() -> int:
    video_folder_raw = _first_env('TRANSCRIBE_VIDEO_FOLDER', 'VIDEO_FOLDER')
    if not video_folder_raw:
        raise RuntimeError('TRANSCRIBE_VIDEO_FOLDER (or VIDEO_FOLDER) is required.')

    video_folder = Path(video_folder_raw)
    if not video_folder.exists():
        raise RuntimeError(f'Video folder does not exist: {video_folder}')

    single_video = _first_env('TRANSCRIBE_SINGLE_VIDEO', default=None)
    videos = _target_videos(video_folder, single_video)
    if not videos:
        raise RuntimeError('No target videos found.')

    whisper_language = _first_env('TRANSCRIBE_WHISPER_LANGUAGE', 'LANGUAGE', default='fr')
    source_lang = _first_env('TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE', default=whisper_language)
    target_lang = _first_env('TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE', default='en')
    translation_model = _first_env('TRANSCRIBE_TRANSLATION_MODEL', default='')
    sidecar_encoding = _first_env(
        'TRANSCRIBE_SIDECAR_SRT_ENCODING',
        'TRANSCRIBE_DUPLICATE_SRT_ENCODING',
        'DUPLICATE_SRT_ENCODING',
        default='utf-8',
    )
    hf_token = _first_env('TRANSCRIBE_HF_TOKEN', 'HF_TOKEN', default='')
    offline_mode = _first_env('TRANSCRIBE_OFFLINE_MODE', default='false')
    merge_output_md = Path(
        _first_env(
            'TRANSCRIBE_SUBTITLE_OUTPUT_MD',
            default=str(video_folder / 'merged_srt_files.md'),
        )
    )

    env_flags = {
        'TRANSCRIBE_RUN_GENERATE_SPANS': _to_bool(
            _first_env('TRANSCRIBE_RUN_GENERATE_SPANS', default='false'),
            False,
        ),
        'TRANSCRIBE_RUN_FILTER_SIDECARS': _to_bool(
            _first_env('TRANSCRIBE_RUN_FILTER_SIDECARS', default='false'),
            False,
        ),
        'TRANSCRIBE_RUN_TRANSCRIPTION': _to_bool(
            _first_env('TRANSCRIBE_RUN_TRANSCRIPTION', default='false'),
            False,
        ),
        'TRANSCRIBE_RUN_TRANSLATION': _to_bool(
            _first_env('TRANSCRIBE_RUN_TRANSLATION', default='false'),
            False,
        ),
        'TRANSCRIBE_RUN_BAKE_SUBTITLES': _to_bool(
            _first_env('TRANSCRIBE_RUN_BAKE_SUBTITLES', default='false'),
            False,
        ),
        'TRANSCRIBE_RUN_SIDECAR_REPLACE': _to_bool(
            _first_env('TRANSCRIBE_RUN_SIDECAR_REPLACE', default='false'),
            False,
        ),
        'TRANSCRIBE_RUN_MERGE': _to_bool(
            _first_env('TRANSCRIBE_RUN_MERGE', default='false'),
            False,
        ),
    }

    resolved_flags = _resolve_interactive_flags(env_flags)
    if resolved_flags is None:
        logger.info('Selection canceled by user. Exiting without running steps.')
        return 0

    run_generate_spans = resolved_flags['TRANSCRIBE_RUN_GENERATE_SPANS']
    run_filter_sidecars = resolved_flags['TRANSCRIBE_RUN_FILTER_SIDECARS']
    run_transcription = resolved_flags['TRANSCRIBE_RUN_TRANSCRIPTION']
    run_translation = resolved_flags['TRANSCRIBE_RUN_TRANSLATION']
    run_bake_subtitles = resolved_flags['TRANSCRIBE_RUN_BAKE_SUBTITLES']
    run_sidecar_replace = resolved_flags['TRANSCRIBE_RUN_SIDECAR_REPLACE']
    run_merge = resolved_flags['TRANSCRIBE_RUN_MERGE']

    force_spans = _to_bool(_first_env('TRANSCRIBE_FORCE_SPANS', default='true'), True)
    force_transcription = _to_bool(
        _first_env('TRANSCRIBE_FORCE_TRANSCRIPTION', default='false'),
        False,
    )
    force_translation = _to_bool(
        _first_env('TRANSCRIBE_FORCE_TRANSLATION', default='true'),
        True,
    )
    force_bake_subtitles = _to_bool(
        _first_env('TRANSCRIBE_FORCE_BAKE_SUBTITLES', default='false'),
        False,
    )
    append_source = _to_bool(
        _first_env('TRANSCRIBE_TRANSLATION_APPEND_SOURCE', default='true'),
        True,
    )

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
        logger.warning('No steps enabled. Enable at least one feature.')
        return 0

    memory = _load_memory()

    common_env = {
        'TRANSCRIBE_VIDEO_FOLDER': str(video_folder),
        'TRANSCRIBE_SINGLE_VIDEO': single_video or '',
        'TRANSCRIBE_WHISPER_LANGUAGE': whisper_language or '',
        'TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE': source_lang or '',
        'TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE': target_lang or '',
        'TRANSCRIBE_TRANSLATION_MODEL': translation_model or '',
        'TRANSCRIBE_SIDECAR_SRT_ENCODING': sidecar_encoding or 'utf-8',
        'TRANSCRIBE_HF_TOKEN': hf_token or '',
        'TRANSCRIBE_OFFLINE_MODE': offline_mode or 'false',
        'TRANSCRIBE_OUTPUT_FOLDER': '',
        'TRANSCRIBE_TRANSLATION_APPEND_SOURCE': 'true' if append_source else 'false',
        'TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION': 'false',
    }

    if run_filter_sidecars:
        missing_spans = [
            video for video in videos if not video.with_suffix('.speech_spans.json').exists()
        ]
        if missing_spans and not run_generate_spans:
            if _ask_with_memory(
                memory,
                'missing_spans_for_filter',
                'Missing speech spans for filter step. Run spans generation first?',
            ):
                run_generate_spans = True
            else:
                run_filter_sidecars = False

    if run_sidecar_replace:
        variant = _variant_for_encoding(sidecar_encoding or 'utf-8')
        missing_variant = [
            video
            for video in videos
            if not (video.parent / video.stem / variant / f'{video.stem}.srt').exists()
        ]
        if missing_variant and not run_translation:
            if _ask_with_memory(
                memory,
                'missing_variant_for_replace',
                f'Missing "{variant}" subtitles for sidecar replace. Run translation first?',
            ):
                run_translation = True
            else:
                run_sidecar_replace = False

    if run_bake_subtitles:
        variant = _variant_for_encoding(sidecar_encoding or 'utf-8')
        missing_variant = [
            video
            for video in videos
            if not (video.parent / video.stem / variant / f'{video.stem}.srt').exists()
        ]
        if missing_variant and not run_translation:
            if _ask_with_memory(
                memory,
                'missing_variant_for_bake',
                f'Missing "{variant}" subtitles for bake step. Run translation first?',
            ):
                run_translation = True
            else:
                run_bake_subtitles = False

    steps: list[tuple[str, str, dict[str, str] | None]] = []

    if run_generate_spans:
        steps.append(
            (
                'generate spans',
                'transcribe',
                {
                    **common_env,
                    'TRANSCRIBE_ENABLE_SPEECH_SPANS': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_ONLY_MODE': 'true',
                    'TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS': 'false',
                    'TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_OVERWRITE': 'true' if force_spans else 'false',
                },
            )
        )

    if run_filter_sidecars:
        steps.append(
            (
                'filter sidecars',
                'transcribe',
                {
                    **common_env,
                    'TRANSCRIBE_ENABLE_SPEECH_SPANS': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_ONLY_MODE': 'true',
                    'TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_OVERWRITE': 'true' if force_spans else 'false',
                },
            )
        )

    if run_transcription:
        if force_transcription:
            logger.info('==> Step: force transcription cleanup')
            for video in videos:
                output_dir = video.parent / video.stem
                if output_dir.exists():
                    for child in output_dir.iterdir():
                        if child.name in {'original', 'stdout.txt'}:
                            if child.is_dir():
                                subprocess.run(
                                    ['cmd', '/c', 'rmdir', '/s', '/q', str(child)],
                                    check=False,
                                )
                            else:
                                child.unlink(missing_ok=True)

        steps.append(
            (
                'transcription',
                'transcribe',
                {
                    **common_env,
                    'TRANSCRIBE_ENABLE_SPEECH_SPANS': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_ONLY_MODE': 'false',
                    'TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS': 'true',
                    'TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING': 'true',
                    'TRANSCRIBE_SPEECH_SPANS_OVERWRITE': 'true' if force_spans else 'false',
                    'TRANSCRIBE_TRANSLATION_ONLY_MODE': 'false',
                    'TRANSCRIBE_ENABLE_TRANSLATION': 'false',
                },
            )
        )

    if run_translation:
        missing_sidecars = [video for video in videos if not video.with_suffix('.srt').exists()]
        if missing_sidecars:
            if not _ask_with_memory(
                memory,
                'missing_sidecars_for_translation',
                'Some sidecar SRT files are missing. Continue translation step anyway?',
            ):
                run_translation = False

    if run_translation:
        steps.append(
            (
                'translation-only from sidecar',
                'transcribe',
                {
                    **common_env,
                    'TRANSCRIBE_ENABLE_SPEECH_SPANS': 'false',
                    'TRANSCRIBE_SPEECH_SPANS_ONLY_MODE': 'false',
                    'TRANSCRIBE_TRANSLATION_ONLY_MODE': 'true',
                    'TRANSCRIBE_TRANSLATION_INPUT': 'sidecar',
                    'TRANSCRIBE_TRANSLATION_OVERWRITE': 'true' if force_translation else 'false',
                },
            )
        )

    if run_merge:
        steps.append(('merge subtitles', 'merge', None))

    if run_sidecar_replace:
        steps.append(('sidecar replace', 'sidecar_replace', None))

    if run_bake_subtitles:
        # Baking must run last.
        steps.append(
            (
                'bake target subtitles into video',
                'transcribe',
                {
                    **common_env,
                    'TRANSCRIBE_ENABLE_SPEECH_SPANS': 'false',
                    'TRANSCRIBE_SPEECH_SPANS_ONLY_MODE': 'false',
                    'TRANSCRIBE_TRANSLATION_ONLY_MODE': 'false',
                    'TRANSCRIBE_ENABLE_TRANSLATION': 'false',
                    'TRANSCRIBE_ENABLE_BAKED_SUBTITLES': 'true',
                    'TRANSCRIBE_BAKE_SUBTITLES_ONLY_MODE': 'true',
                    'TRANSCRIBE_BAKE_SUBTITLES_OVERWRITE': 'true'
                    if force_bake_subtitles
                    else 'false',
                },
            )
        )

    total_steps = len(steps)
    for idx, (step_name, step_kind, step_env) in enumerate(steps, start=1):
        if step_kind == 'transcribe':
            if step_env is None:
                raise RuntimeError(f'Missing env for step: {step_name}')
            _run_transcribe_step(
                name=step_name,
                env_overrides=step_env,
                step_index=idx,
                total_steps=total_steps,
            )
        elif step_kind == 'merge':
            _run_merge_step(
                video_folder=video_folder,
                sidecar_encoding=sidecar_encoding or 'utf-8',
                output_md=merge_output_md,
                step_index=idx,
                total_steps=total_steps,
            )
        elif step_kind == 'sidecar_replace':
            _run_sidecar_replace_step(
                video_folder=video_folder,
                sidecar_encoding=sidecar_encoding or 'utf-8',
                single_video=single_video,
                step_index=idx,
                total_steps=total_steps,
            )

    logger.info('Master pipeline finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
