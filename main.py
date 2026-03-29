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

from explicit_content_cut import (
    build_explicit_cut_plan,
    load_explicit_cut_config_from_env,
    write_explicit_cut_plan_report,
)
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
    ('TRANSCRIBE_RUN_CUT_EXPLICIT_CONTENT', 'Cut explicit content'),
    ('TRANSCRIBE_RUN_TRANSLATION', 'Run translation from sidecar'),
    ('TRANSCRIBE_RUN_MERGE', 'Merge subtitles to Markdown'),
    ('TRANSCRIBE_RUN_SIDECAR_REPLACE', 'Replace sidecar from variant'),
    ('TRANSCRIBE_RUN_BAKE_SUBTITLES', 'Bake subtitles into video'),
]
TRANSLATING_LINES_RE = re.compile(
    r'Translating\s+(\d+)\s+subtitle block\(s\)\s+from\s+(.+?)\s+in\s+(\d+)\s+chunk\(es\)'
)
TRANSLATION_BATCH_RE = re.compile(
    r'Translation batch\s+(\d+)/(\d+)\s+\((\d+)/(\d+)\s+blocks,\s+elapsed\s+([0-9.]+)s,\s+eta\s+([0-9.]+s)\)'
)
MODEL_LOADING_RE = re.compile(r'Loading translation model:\s+(.+?)\s+\(offline=(True|False)\)')
MODEL_LOADED_RE = re.compile(r'Translation model loaded in\s+([0-9.]+)s')
TRANSLATION_FINISHED_RE = re.compile(r'Subtitle translation finished in\s+([0-9.]+)s')
FEATURE_DETAILS = {
    'TRANSCRIBE_RUN_CUT_EXPLICIT_CONTENT': (
        'Generate a cut plan from subtitles, then apply only after approval.',
        '~2-15+ min/video',
    ),
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


def _language_script_group(language: str) -> str | None:
    normalized = language.lower()
    if normalized.startswith(('ru', 'uk', 'be', 'bg', 'sr')):
        return 'cyrillic'
    if normalized.startswith('el'):
        return 'greek'
    if normalized.startswith(('ar', 'fa', 'ur')):
        return 'arabic'
    if normalized.startswith('he'):
        return 'hebrew'
    return None


def _contains_target_script(text: str, language: str) -> bool:
    script_group = _language_script_group(language)
    if script_group == 'cyrillic':
        return bool(re.search(r'[А-Яа-яЁёІіЇїЄєЎў]', text))
    if script_group == 'greek':
        return bool(re.search(r'[Α-Ωα-ω]', text))
    if script_group == 'arabic':
        return bool(re.search(r'[\u0600-\u06FF]', text))
    if script_group == 'hebrew':
        return bool(re.search(r'[\u0590-\u05FF]', text))
    return False


def _sidecar_contains_target_language(
    path: Path,
    target_language: str,
    max_lines: int = 200,
) -> tuple[bool, int]:
    if not path.exists():
        return False, 0

    total_checked = 0
    target_hits = 0
    for encoding in ('utf-8', 'utf-8-sig', 'windows-1251', 'cp1251'):
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        return False, 0

    for line in lines:
        stripped = line.strip()
        if not stripped or '-->' in stripped or stripped.isdigit():
            continue
        total_checked += 1
        if _contains_target_script(stripped, target_language):
            target_hits += 1
        if total_checked >= max_lines:
            break

    return target_hits > 0, target_hits


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


def _ask_yes_no(question: str) -> bool:
    if not sys.stdin.isatty():
        raise RuntimeError(f'Need user input for "{question}" in non-interactive mode.')

    while True:
        answer = input(f'{question} [y/n]: ').strip().lower()
        if answer in {'y', 'yes'}:
            return True
        if answer in {'n', 'no'}:
            return False


def _decision_policy_env_name(key: str) -> str:
    return f"TRANSCRIBE_POLICY_{key.upper()}"


def _resolve_yes_no_decision(
    key: str,
    question: str,
    default_policy: str = 'ask',
) -> bool:
    env_name = _decision_policy_env_name(key)
    policy = (_first_env(env_name, default=default_policy) or default_policy).strip().lower()

    if policy in {'y', 'yes', 'true', '1', 'on'}:
        logger.info('[policy:%s] yes -> %s', env_name, question)
        return True
    if policy in {'n', 'no', 'false', '0', 'off'}:
        logger.info('[policy:%s] no -> %s', env_name, question)
        return False
    if policy != 'ask':
        raise RuntimeError(
            f'Invalid {env_name} value: {policy!r}. Use ask, yes, or no.'
        )
    if not sys.stdin.isatty():
        raise RuntimeError(
            f'Need user input for "{question}" in non-interactive mode. '
            f'Set {env_name}=yes or {env_name}=no to choose a policy.'
        )
    return _ask_yes_no(question)


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
        process = subprocess.Popen([sys.executable, 'transcribe.py'], env=env)
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            logger.warning('Interrupted. Stopping current step...')
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        if return_code != 0:
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
    stage_label = 'starting...'
    translation_total_batches = 0

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

        def _feature_stats(videos_left: int | None = None) -> str:
            if videos_left is None:
                return f'stage: {stage_label}'
            return f'stage: {stage_label} | videos left: {videos_left}'

        def _video_stats(
            elapsed_override: float | None = None,
            eta_text: str | None = None,
            videos_left: int | None = None,
        ) -> str:
            elapsed = time.perf_counter() - start_time if elapsed_override is None else elapsed_override
            parts = [f'elapsed: {elapsed/60:.1f}m', f'eta: {eta_text or "n/a"}']
            if videos_left is not None:
                parts.append(f'videos left: {videos_left}')
            parts.append(f'stage: {stage_label}')
            return ' | '.join(parts)

        try:
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
                    progress.update(
                        feature_task,
                        total=total_videos,
                        completed=completed_videos,
                        stats=_feature_stats(videos_left),
                    )
                    progress.update(
                        video_task,
                        description=f'Video {current_video}/{total_videos}: {current_video_name}',
                        total=total_videos,
                        completed=completed_videos,
                        stats=_video_stats(
                            elapsed_override=elapsed,
                            eta_text=eta_text,
                            videos_left=videos_left,
                        ),
                    )
                    continue

                match = MODEL_LOADING_RE.search(line)
                if match:
                    stage_label = f'loading model: {match.group(1)}'
                    progress.update(feature_task, stats=_feature_stats(total_videos - max(current_video - 1, 0) if total_videos else None))
                    progress.update(video_task, stats=_video_stats(videos_left=(total_videos - max(current_video - 1, 0)) if total_videos else None))
                    continue

                match = MODEL_LOADED_RE.search(line)
                if match:
                    stage_label = f'model loaded in {match.group(1)}s'
                    progress.update(feature_task, stats=_feature_stats(total_videos - max(current_video - 1, 0) if total_videos else None))
                    progress.update(video_task, stats=_video_stats(videos_left=(total_videos - max(current_video - 1, 0)) if total_videos else None))
                    continue

                match = TRANSLATING_LINES_RE.search(line)
                if match:
                    total_lines = int(match.group(1))
                    translation_total_batches = int(match.group(3))
                    stage_label = f'translating {total_lines} subtitle blocks'
                    progress.update(
                        video_task,
                        total=max(translation_total_batches, 1),
                        completed=0,
                        stats=_video_stats(videos_left=(total_videos - max(current_video - 1, 0)) if total_videos else None),
                    )
                    progress.update(feature_task, stats=_feature_stats((total_videos - max(current_video - 1, 0)) if total_videos else None))
                    continue

                match = TRANSLATION_BATCH_RE.search(line)
                if match:
                    batch_index = int(match.group(1))
                    total_batches = int(match.group(2))
                    processed_lines = int(match.group(3))
                    total_lines = int(match.group(4))
                    elapsed_seconds = float(match.group(5))
                    eta_text = match.group(6)
                    stage_label = f'translating {processed_lines}/{total_lines} blocks'
                    progress.update(
                        video_task,
                        total=max(total_batches, 1),
                        completed=batch_index,
                        stats=_video_stats(
                            elapsed_override=elapsed_seconds,
                            eta_text=eta_text,
                            videos_left=(total_videos - max(current_video - 1, 0)) if total_videos else None,
                        ),
                    )
                    progress.update(feature_task, stats=_feature_stats((total_videos - max(current_video - 1, 0)) if total_videos else None))
                    continue

                match = TRANSLATION_FINISHED_RE.search(line)
                if match:
                    stage_label = f'translation finished in {match.group(1)}s'
                    progress.update(
                        video_task,
                        total=max(translation_total_batches, 1),
                        completed=max(translation_total_batches, 1),
                        stats=_video_stats(
                            elapsed_override=float(match.group(1)),
                            eta_text='0.0s',
                            videos_left=(total_videos - max(current_video - 1, 0)) if total_videos else None,
                        ),
                    )
                    progress.update(feature_task, stats=_feature_stats((total_videos - max(current_video - 1, 0)) if total_videos else None))
                    continue

                if any(token in line for token in ('ERROR', 'Traceback', 'Failed')):
                    console.print(f'[red]{line}[/red]')
        except KeyboardInterrupt:
            logger.warning('Interrupted. Stopping current step...')
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()

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


def _run_explicit_cut_step(
    env_overrides: dict[str, str],
    step_index: int,
    total_steps: int,
) -> None:
    env = os.environ.copy()
    env.update(env_overrides)
    env['PYTHONUNBUFFERED'] = '1'
    logger.info('==> Step %d/%d: cut explicit content', step_index, total_steps)
    process = subprocess.run([sys.executable, 'explicit_content_cut.py'], env=env, check=False)
    if process.returncode != 0:
        raise RuntimeError('Step failed: cut explicit content')


def _explicit_cut_report_path(video: Path) -> Path:
    return video.parent / f'{video.stem}.explicit_cut_report.json'


def _generate_explicit_cut_reports(
    env_overrides: dict[str, str],
    videos: list[Path],
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    config = load_explicit_cut_config_from_env()
    for video in videos:
        try:
            plan = build_explicit_cut_plan(video, config)
        except FileNotFoundError:
            continue
        write_explicit_cut_plan_report(video, plan)
        reports.append(plan.to_dict())
    return reports


def _preflight_explicit_cut_plan(
    run_cut_explicit_content: bool,
    common_env: dict[str, str],
    videos: list[Path],
) -> bool:
    if not run_cut_explicit_content:
        return False

    reports = _generate_explicit_cut_reports(common_env, videos)
    planned_reports = [
        report for report in reports if float(report.get('cut_duration_sec', 0.0)) > 0.0
    ]
    if not planned_reports:
        logger.info('Explicit-cut preflight found no matching scenes to remove.')
        return False

    summary_items: list[str] = []
    for report in planned_reports[:3]:
        cut_spans = report.get('cut_spans', [])
        first_span = cut_spans[0] if cut_spans else None
        if isinstance(first_span, dict):
            summary_items.append(
                f"{report.get('video_file')} ({len(cut_spans)} cut(s), "
                f"{report.get('cut_duration_sec')}s total, "
                f"first: {first_span.get('start')}..{first_span.get('end')})"
            )
        else:
            summary_items.append(
                f"{report.get('video_file')} ({report.get('cut_duration_sec')}s total)"
            )
    if len(planned_reports) > 3:
        summary_items.append(f'+{len(planned_reports) - 3} more')

    question = (
        'Explicit-cut preflight report is ready. Approve applying the planned cuts? '
        + '; '.join(summary_items)
    )
    return _resolve_yes_no_decision('approve_explicit_cut_plan', question)


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


def _preflight_sidecar_translation_source(
    videos: list[Path],
    target_language: str,
    run_translation: bool,
) -> bool:
    if not run_translation:
        return run_translation

    suspicious_sources: list[str] = []
    for video in videos:
        sidecar = video.with_suffix('.srt')
        contains_target, target_hits = _sidecar_contains_target_language(sidecar, target_language)
        if contains_target:
            suspicious_sources.append(f'{sidecar.name} ({target_hits} target-language line(s) detected)')

    if not suspicious_sources:
        return run_translation

    preview = '; '.join(suspicious_sources[:3])
    if len(suspicious_sources) > 3:
        preview += f'; +{len(suspicious_sources) - 3} more'
    question = (
        'Sidecar SRT source already appears to contain target-language subtitle lines '
        f'for {target_language}. Re-translate anyway? {preview}'
    )
    return _resolve_yes_no_decision('mixed_language_sidecar_source', question)


def _resolve_bake_subtitle_source(
    videos: list[Path],
    sidecar_encoding: str,
    run_translation: bool,
) -> tuple[bool, str]:
    variant = _variant_for_encoding(sidecar_encoding)
    missing_variant = [
        video
        for video in videos
        if not (video.parent / video.stem / variant / f'{video.stem}.srt').exists()
    ]
    if not missing_variant or run_translation:
        return run_translation, 'target'

    if _resolve_yes_no_decision(
        'missing_variant_for_bake',
        f'Missing "{variant}" subtitles for bake step. Run translation first?',
    ):
        return True, 'target'

    available_sidecars = [video for video in videos if video.with_suffix('.srt').exists()]
    if not available_sidecars:
        logger.info('Bake fallback is unavailable because no root sidecar SRT files were found.')
        return False, 'target'

    preview = ', '.join(video.name for video in available_sidecars[:3])
    if len(available_sidecars) < len(videos):
        preview += f'; {len(videos) - len(available_sidecars)} video(s) still missing sidecar SRT'
    if _resolve_yes_no_decision(
        'fallback_sidecar_for_bake',
        f'Bake from existing sidecar SRT instead? {preview}',
    ):
        logger.info('Bake step will use existing root sidecar SRT files as the subtitle source.')
        return False, 'sidecar'

    return False, 'target'


def _cancel_translation_dependent_steps(
    translation_requested: bool,
    run_translation: bool,
    run_sidecar_replace: bool,
    run_bake_subtitles: bool,
) -> tuple[bool, bool]:
    if translation_requested and not run_translation:
        if run_sidecar_replace:
            logger.info(
                'Canceling sidecar replace because the selected translation step was declined during preflight.'
            )
            run_sidecar_replace = False
        if run_bake_subtitles:
            logger.info(
                'Canceling bake subtitles because the selected translation step was declined during preflight.'
            )
            run_bake_subtitles = False
    return run_sidecar_replace, run_bake_subtitles


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
        'TRANSCRIBE_RUN_CUT_EXPLICIT_CONTENT': _to_bool(
            _first_env('TRANSCRIBE_RUN_CUT_EXPLICIT_CONTENT', default='false'),
            False,
        ),
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

    run_cut_explicit_content = resolved_flags['TRANSCRIBE_RUN_CUT_EXPLICIT_CONTENT']
    run_generate_spans = resolved_flags['TRANSCRIBE_RUN_GENERATE_SPANS']
    run_filter_sidecars = resolved_flags['TRANSCRIBE_RUN_FILTER_SIDECARS']
    run_transcription = resolved_flags['TRANSCRIBE_RUN_TRANSCRIPTION']
    run_translation = resolved_flags['TRANSCRIBE_RUN_TRANSLATION']
    run_bake_subtitles = resolved_flags['TRANSCRIBE_RUN_BAKE_SUBTITLES']
    run_sidecar_replace = resolved_flags['TRANSCRIBE_RUN_SIDECAR_REPLACE']
    run_merge = resolved_flags['TRANSCRIBE_RUN_MERGE']
    translation_requested = run_translation

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
    bake_subtitle_source = 'target'

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
            if _resolve_yes_no_decision(
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
            if _resolve_yes_no_decision(
                'missing_variant_for_replace',
                f'Missing "{variant}" subtitles for sidecar replace. Run translation first?',
            ):
                run_translation = True
            else:
                run_sidecar_replace = False

    if run_bake_subtitles:
        run_translation, bake_subtitle_source = _resolve_bake_subtitle_source(
            videos=videos,
            sidecar_encoding=sidecar_encoding or 'utf-8',
            run_translation=run_translation,
        )
        if bake_subtitle_source != 'sidecar':
            variant = _variant_for_encoding(sidecar_encoding or 'utf-8')
            missing_variant = [
                video
                for video in videos
                if not (video.parent / video.stem / variant / f'{video.stem}.srt').exists()
            ]
            if missing_variant and not run_translation:
                run_bake_subtitles = False

    if run_cut_explicit_content:
        missing_sidecars_for_cut = [
            video for video in videos if not video.with_suffix('.srt').exists()
        ]
        if missing_sidecars_for_cut and run_transcription:
            logger.warning(
                'Explicit-cut preflight requires existing sidecar subtitles before the run. '
                'Run transcription first, then run explicit cut in a follow-up pass.'
            )
            run_cut_explicit_content = False
        elif missing_sidecars_for_cut:
            if not _resolve_yes_no_decision(
                'missing_sidecars_for_explicit_cut',
                'Some sidecar SRT files are missing. Continue explicit-content cut anyway?',
            ):
                run_cut_explicit_content = False
        else:
            run_cut_explicit_content = _preflight_explicit_cut_plan(
                run_cut_explicit_content=run_cut_explicit_content,
                common_env=common_env,
                videos=videos,
            )

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

    if run_cut_explicit_content:
        steps.append(
            (
                'cut explicit content',
                'explicit_cut',
                {
                    **common_env,
                },
            )
        )

    if run_translation:
        missing_sidecars = [video for video in videos if not video.with_suffix('.srt').exists()]
        if missing_sidecars:
            if not _resolve_yes_no_decision(
                'missing_sidecars_for_translation',
                'Some sidecar SRT files are missing. Continue translation step anyway?',
            ):
                run_translation = False

    run_translation = _preflight_sidecar_translation_source(
        videos=videos,
        target_language=target_lang or '',
        run_translation=run_translation,
    )
    run_sidecar_replace, run_bake_subtitles = _cancel_translation_dependent_steps(
        translation_requested=translation_requested,
        run_translation=run_translation,
        run_sidecar_replace=run_sidecar_replace,
        run_bake_subtitles=run_bake_subtitles,
    )

    logger.info(
        'Final step flags after preflight: cut_explicit_content=%s, generate_spans=%s, filter_sidecars=%s, '
        'transcription=%s, translation=%s, bake_subtitles=%s, '
        'sidecar_replace=%s, merge=%s',
        run_cut_explicit_content,
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
            run_cut_explicit_content,
            run_generate_spans,
            run_filter_sidecars,
            run_transcription,
            run_translation,
            run_bake_subtitles,
            run_sidecar_replace,
            run_merge,
        ]
    ):
        logger.warning('No steps remain after preflight checks. Exiting without running steps.')
        return 0

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
                    'TRANSCRIBE_BAKE_SUBTITLE_SOURCE': bake_subtitle_source,
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
        elif step_kind == 'explicit_cut':
            if step_env is None:
                raise RuntimeError(f'Missing env for step: {step_name}')
            _run_explicit_cut_step(
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
