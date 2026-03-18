"""Combine subtitle files into one Markdown document."""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

TIMECODE_LINE = re.compile(
    r'^\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*$'
)
VARIANT_PRIORITY = {
    'translated_utf8': 1,
    'translated_windows1251': 2,
    'original': 3,
}


@dataclass
class MergeReport:
    files_discovered: int = 0
    files_merged: int = 0
    section_titles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != '':
            return value.strip()
    return default


def _parse_encodings(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default

    encodings = [part.strip() for part in value.split(',') if part.strip()]
    return encodings if encodings else default


def _is_index_line(line: str) -> bool:
    return line.strip().isdigit()


def _is_timecode_line(line: str) -> bool:
    return TIMECODE_LINE.match(line) is not None


def _extract_subtitle_blocks(content: str) -> list[str]:
    raw_blocks = re.split(r'\r?\n\s*\r?\n', content.strip())
    cleaned_blocks: list[str] = []

    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines()]
        text_lines = [
            line
            for line in lines
            if line and not _is_index_line(line) and not _is_timecode_line(line)
        ]

        if text_lines:
            # Use Markdown hard line breaks to keep multiline subtitle cues visible.
            cleaned_blocks.append('  \n'.join(text_lines))

    return cleaned_blocks


def _build_section_title(source_dir: Path, subtitle_file: Path) -> str:
    relative_parent = subtitle_file.parent.relative_to(source_dir)
    parent_name = relative_parent.as_posix() if relative_parent.parts else source_dir.name
    return f'{parent_name} / {subtitle_file.name}'


def _episode_key(source_dir: Path, subtitle_file: Path) -> str:
    relative = subtitle_file.relative_to(source_dir)
    parent_parts = relative.parts[:-1]

    if parent_parts and parent_parts[-1] in VARIANT_PRIORITY:
        episode_parts = parent_parts[:-1]
    else:
        episode_parts = parent_parts

    if not episode_parts:
        return relative.stem

    return '/'.join(episode_parts)


def _file_rank(source_dir: Path, subtitle_file: Path) -> tuple[int, int, str]:
    relative = subtitle_file.relative_to(source_dir)
    parent_parts = relative.parts[:-1]

    if parent_parts and parent_parts[-1] in VARIANT_PRIORITY:
        variant_rank = VARIANT_PRIORITY[parent_parts[-1]]
    else:
        variant_rank = 0

    return variant_rank, len(relative.parts), relative.as_posix()


def _select_preferred_files(
    source_dir: Path,
    subtitle_files: list[Path],
    report: MergeReport,
) -> list[Path]:
    best_per_episode: dict[str, Path] = {}
    ranked_per_episode: dict[str, tuple[int, int, str]] = {}
    all_per_episode: dict[str, list[Path]] = {}

    for subtitle_file in subtitle_files:
        episode = _episode_key(source_dir, subtitle_file)
        rank = _file_rank(source_dir, subtitle_file)
        all_per_episode.setdefault(episode, []).append(subtitle_file)

        if episode not in best_per_episode or rank < ranked_per_episode[episode]:
            best_per_episode[episode] = subtitle_file
            ranked_per_episode[episode] = rank

    selected_files = []
    for episode in sorted(best_per_episode):
        selected = best_per_episode[episode]
        selected_files.append(selected)
        candidates = all_per_episode[episode]
        if len(candidates) > 1:
            skipped_count = len(candidates) - 1
            report.warnings.append(
                f'Found {len(candidates)} subtitle variants for "{episode}". '
                f'Using "{selected.relative_to(source_dir)}", skipped {skipped_count}.'
            )

    return selected_files


def _collect_sections(
    source_dir: Path,
    pattern: str,
    source_encoding: str,
    source_fallback_encodings: list[str],
) -> tuple[list[str], MergeReport]:
    subtitle_files = sorted(path for path in source_dir.rglob(pattern) if path.is_file())

    report = MergeReport(files_discovered=len(subtitle_files))
    subtitle_files = _select_preferred_files(source_dir, subtitle_files, report)
    sections: list[str] = []

    for subtitle_file in subtitle_files:
        read_order = [source_encoding, *source_fallback_encodings]
        # Keep order while removing duplicates
        ordered_unique = list(dict.fromkeys(read_order))

        content: str | None = None
        decode_errors: list[str] = []

        for encoding in ordered_unique:
            try:
                content = subtitle_file.read_text(
                    encoding=encoding,
                    errors='strict',
                )
                if encoding != source_encoding:
                    report.warnings.append(
                        f'Used fallback encoding "{encoding}" for {subtitle_file}'
                    )
                break
            except UnicodeDecodeError as error:
                decode_errors.append(f'{encoding}: {error}')
                continue
            except OSError as error:
                report.errors.append(f'Failed to read {subtitle_file}: {error}')
                content = None
                break

        if content is None:
            report.errors.append(
                f'Failed to decode {subtitle_file}. Tried encodings: '
                f'{", ".join(ordered_unique)}. Details: {" | ".join(decode_errors)}'
            )
            continue

        blocks = _extract_subtitle_blocks(content)

        if not blocks:
            report.warnings.append(f'Skipping empty subtitle file: {subtitle_file}')
            continue

        title = _build_section_title(source_dir, subtitle_file)
        body = '\n\n'.join(blocks)
        sections.append(f'## {title}\n\n{body}')
        report.section_titles.append(title)
        report.files_merged += 1

    return sections, report


def _log_merge_report(report: MergeReport) -> None:
    logger.info(
        'Merge finished. Files discovered: %d. Files merged: %d.',
        report.files_discovered,
        report.files_merged,
    )

    logger.info('Warnings (%d):', len(report.warnings))
    for warning in report.warnings:
        logger.warning('  - %s', warning)

    logger.info('Errors (%d):', len(report.errors))
    for error in report.errors:
        logger.error('  - %s', error)


def merge_subtitles(
    source_dir: Path,
    output_md: Path,
    pattern: str,
    source_encoding: str,
    output_encoding: str,
    source_fallback_encodings: list[str] | None = None,
) -> MergeReport:
    if source_fallback_encodings is None:
        source_fallback_encodings = ['utf-8', 'utf-8-sig', 'windows-1251', 'cp1251']

    sections, report = _collect_sections(
        source_dir=source_dir,
        pattern=pattern,
        source_encoding=source_encoding,
        source_fallback_encodings=source_fallback_encodings,
    )

    if not sections:
        report.warnings.append(
            f'No subtitle files matched pattern {pattern} in {source_dir}'
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text('', encoding=output_encoding)
        _log_merge_report(report)
        return report

    document = '# Combined Subtitles\n\n' + '\n\n---\n\n'.join(sections) + '\n'

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(document, encoding=output_encoding, errors='replace')
    logger.info(
        'Saved Markdown file: %s (source encoding: %s, output encoding: %s)',
        output_md,
        source_encoding,
        output_encoding,
    )

    _log_merge_report(report)
    return report


def main() -> int:
    source_dir_raw = _first_env('TRANSCRIBE_SUBTITLE_SOURCE_DIR')
    if not source_dir_raw:
        raise RuntimeError('TRANSCRIBE_SUBTITLE_SOURCE_DIR is not set in .env')

    output_md_raw = _first_env('TRANSCRIBE_SUBTITLE_OUTPUT_MD')
    if not output_md_raw:
        raise RuntimeError('TRANSCRIBE_SUBTITLE_OUTPUT_MD is not set in .env')

    subtitle_glob = _first_env('TRANSCRIBE_SUBTITLE_GLOB', default='*.srt')
    default_srt_encoding = _first_env(
        'TRANSCRIBE_SIDECAR_SRT_ENCODING',
        'TRANSCRIBE_DUPLICATE_SRT_ENCODING',
        'DUPLICATE_SRT_ENCODING',
        default='utf-8',
    )
    subtitle_source_encoding = _first_env(
        'TRANSCRIBE_SUBTITLE_SOURCE_ENCODING',
        default=default_srt_encoding,
    )
    subtitle_source_fallback_encodings = _parse_encodings(
        _first_env('TRANSCRIBE_SUBTITLE_SOURCE_ENCODING_FALLBACKS', default=None),
        default=['utf-8', 'utf-8-sig', 'windows-1251', 'cp1251'],
    )
    subtitle_output_encoding = _first_env(
        'TRANSCRIBE_SUBTITLE_OUTPUT_ENCODING',
        default=default_srt_encoding,
    )

    source_dir = Path(source_dir_raw)
    output_md = Path(output_md_raw)

    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(f'Subtitle source directory does not exist: {source_dir}')

    report = merge_subtitles(
        source_dir=source_dir,
        output_md=output_md,
        pattern=subtitle_glob,
        source_encoding=subtitle_source_encoding,
        output_encoding=subtitle_output_encoding,
        source_fallback_encodings=subtitle_source_fallback_encodings,
    )
    return 1 if report.errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
