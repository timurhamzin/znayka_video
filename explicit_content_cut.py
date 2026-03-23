"""Cut explicit-content scenes from videos using sidecar subtitles as markers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class SRTBlock:
    start: float
    end: float
    lines: list[str]


@dataclass
class TimeSpan:
    start: float
    end: float


@dataclass(frozen=True)
class ExplicitCutConfig:
    sidecar_encoding: str
    keywords: list[str]
    margin_before_sec: float
    margin_after_sec: float
    group_gap_sec: float
    scene_gap_sec: float
    max_extend_before_sec: float
    max_extend_after_sec: float
    preset: str
    crf: str
    frame_verification_backend: str = "off"
    frame_interval_sec: float = 2.0
    frame_nsfw_threshold: float = 0.7
    frame_min_positive_ratio: float = 0.2
    frame_max_samples_per_span: int = 12
    force: bool = False


@dataclass(frozen=True)
class ExplicitCutPlan:
    video_file: str
    sidecar_file: str
    sidecar_encoding_used: str
    keywords: list[str]
    matched_block_indexes: list[int]
    cut_spans: list[TimeSpan]
    original_duration_sec: float
    cut_duration_sec: float
    result_duration_sec: float
    frame_verification_backend: str = "off"
    frame_verification_passed: bool | None = None
    frame_verification_summary: str | None = None
    frame_verification_samples: list[dict[str, object]] | None = None
    applied: bool = False
    backup_dir: str | None = None
    removed_stale_outputs: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "video_file": self.video_file,
            "sidecar_file": self.sidecar_file,
            "sidecar_encoding_used": self.sidecar_encoding_used,
            "keywords": self.keywords,
            "matched_block_indexes": self.matched_block_indexes,
            "cut_spans": [
                {
                    "start": round(span.start, 3),
                    "end": round(span.end, 3),
                    "duration_sec": round(span.end - span.start, 3),
                }
                for span in self.cut_spans
            ],
            "original_duration_sec": round(self.original_duration_sec, 3),
            "cut_duration_sec": round(self.cut_duration_sec, 3),
            "result_duration_sec": round(self.result_duration_sec, 3),
            "frame_verification_backend": self.frame_verification_backend,
            "frame_verification_passed": self.frame_verification_passed,
            "frame_verification_summary": self.frame_verification_summary,
            "frame_verification_samples": self.frame_verification_samples or [],
            "backup_dir": self.backup_dir,
            "removed_stale_outputs": self.removed_stale_outputs or [],
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ExplicitCutPlan":
        raw_spans = payload.get("cut_spans", [])
        cut_spans = [
            TimeSpan(start=float(item["start"]), end=float(item["end"]))
            for item in raw_spans
            if isinstance(item, dict)
        ]
        return cls(
            video_file=str(payload.get("video_file", "")),
            sidecar_file=str(payload.get("sidecar_file", "")),
            sidecar_encoding_used=str(payload.get("sidecar_encoding_used", "utf-8")),
            keywords=[str(item) for item in payload.get("keywords", [])],
            matched_block_indexes=[int(item) for item in payload.get("matched_block_indexes", [])],
            cut_spans=cut_spans,
            original_duration_sec=float(payload.get("original_duration_sec", 0.0)),
            cut_duration_sec=float(payload.get("cut_duration_sec", 0.0)),
            result_duration_sec=float(payload.get("result_duration_sec", 0.0)),
            frame_verification_backend=str(payload.get("frame_verification_backend", "off")),
            frame_verification_passed=(
                bool(payload["frame_verification_passed"])
                if payload.get("frame_verification_passed") is not None
                else None
            ),
            frame_verification_summary=(
                str(payload["frame_verification_summary"])
                if payload.get("frame_verification_summary")
                else None
            ),
            frame_verification_samples=[
                item
                for item in payload.get("frame_verification_samples", [])
                if isinstance(item, dict)
            ],
            applied=bool(payload.get("applied", False)),
            backup_dir=str(payload["backup_dir"]) if payload.get("backup_dir") else None,
            removed_stale_outputs=[str(item) for item in payload.get("removed_stale_outputs", [])],
        )


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _seconds_to_srt_time(value: float) -> str:
    clamped = max(0.0, value)
    total_ms = int(round(clamped * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    seconds, ms = divmod(rem_ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def _parse_srt_time(value: str) -> float:
    hours, minutes, seconds_ms = value.split(":")
    seconds, milliseconds = seconds_ms.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def _parse_srt(content: str) -> list[SRTBlock]:
    blocks: list[SRTBlock] = []
    chunks = [chunk for chunk in content.replace("\r\n", "\n").split("\n\n") if chunk.strip()]
    for chunk in chunks:
        lines = chunk.splitlines()
        if len(lines) < 2:
            continue
        line_index = 0
        if lines[0].strip().isdigit():
            line_index = 1
        if line_index >= len(lines) or "-->" not in lines[line_index]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[line_index].split("-->", maxsplit=1)]
        text_lines = [line.strip() for line in lines[line_index + 1:] if line.strip()]
        if not text_lines:
            continue
        blocks.append(
            SRTBlock(
                start=_parse_srt_time(start_raw),
                end=_parse_srt_time(end_raw),
                lines=text_lines,
            )
        )
    return blocks


def _serialize_srt(blocks: list[SRTBlock]) -> str:
    rendered: list[str] = []
    for index, block in enumerate(blocks, start=1):
        rendered.append(str(index))
        rendered.append(
            f"{_seconds_to_srt_time(block.start)} --> {_seconds_to_srt_time(block.end)}"
        )
        rendered.extend(block.lines)
        rendered.append("")
    return "\n".join(rendered).rstrip() + "\n" if rendered else ""


def _read_text_with_fallbacks(path: Path, primary_encoding: str) -> tuple[str, str]:
    encodings = [primary_encoding, "utf-8", "utf-8-sig", "windows-1251", "cp1251"]
    tried: list[str] = []
    for encoding in encodings:
        if encoding in tried:
            continue
        tried.append(encoding)
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Failed to decode {path} using: {', '.join(tried)}",
    )


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


def _merge_spans(spans: list[TimeSpan], gap_sec: float) -> list[TimeSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: item.start)
    merged: list[TimeSpan] = [ordered[0]]
    for span in ordered[1:]:
        current = merged[-1]
        if span.start <= current.end + gap_sec:
            current.end = max(current.end, span.end)
        else:
            merged.append(TimeSpan(start=span.start, end=span.end))
    return merged


def _overlaps(block: SRTBlock, span: TimeSpan) -> bool:
    return max(block.start, span.start) < min(block.end, span.end)


def _removed_before(value: float, cut_spans: list[TimeSpan]) -> float:
    removed = 0.0
    for span in cut_spans:
        if value >= span.end:
            removed += span.end - span.start
        elif value > span.start:
            removed += value - span.start
            break
        else:
            break
    return removed


def _safe_name(keyword: str) -> str:
    return keyword.strip().lower()


def _default_keywords() -> list[str]:
    return [
        "very sexy",
        "black lace bra",
        "panties",
        "take my bra off",
        "nipples",
        "rubbing my breasts",
        "my breasts",
        "moaning",
        "inside my panties",
        "completely naked",
        "black stockings",
        "high heels",
    ]


def _explicit_keywords() -> list[str]:
    raw = _first_env("TRANSCRIBE_EXPLICIT_CONTENT_KEYWORDS", default="")
    if raw:
        return [_safe_name(item) for item in raw.split(",") if item.strip()]
    return _default_keywords()


def load_explicit_cut_config_from_env() -> ExplicitCutConfig:
    sidecar_encoding = _first_env(
        "TRANSCRIBE_SIDECAR_SRT_ENCODING",
        "TRANSCRIBE_DUPLICATE_SRT_ENCODING",
        "DUPLICATE_SRT_ENCODING",
        default="utf-8",
    ) or "utf-8"
    return ExplicitCutConfig(
        sidecar_encoding=sidecar_encoding,
        keywords=_explicit_keywords(),
        margin_before_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_MARGIN_BEFORE_SEC", default="0.5")
        ),
        margin_after_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_MARGIN_AFTER_SEC", default="0.5")
        ),
        group_gap_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_GROUP_GAP_SEC", default="20")
        ),
        scene_gap_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_SCENE_GAP_SEC", default="5")
        ),
        max_extend_before_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_MAX_EXTEND_BEFORE_SEC", default="5")
        ),
        max_extend_after_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_MAX_EXTEND_AFTER_SEC", default="45")
        ),
        preset=_first_env("TRANSCRIBE_EXPLICIT_CUT_VIDEO_PRESET", default="veryfast")
        or "veryfast",
        crf=_first_env("TRANSCRIBE_EXPLICIT_CUT_VIDEO_CRF", default="20") or "20",
        frame_verification_backend=(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_FRAME_BACKEND", default="off") or "off"
        ).strip().lower(),
        frame_interval_sec=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_FRAME_INTERVAL_SEC", default="2.0")
        ),
        frame_nsfw_threshold=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_FRAME_NSFW_THRESHOLD", default="0.7")
        ),
        frame_min_positive_ratio=float(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_FRAME_MIN_POSITIVE_RATIO", default="0.2")
        ),
        frame_max_samples_per_span=int(
            _first_env("TRANSCRIBE_EXPLICIT_CUT_FRAME_MAX_SAMPLES_PER_SPAN", default="12")
        ),
        force=_parse_bool(_first_env("TRANSCRIBE_FORCE_CUT_EXPLICIT_CONTENT", default="false")),
    )


def _detect_explicit_spans(
    blocks: list[SRTBlock],
    keywords: list[str],
    margin_before_sec: float,
    margin_after_sec: float,
    group_gap_sec: float,
    scene_gap_sec: float,
    max_extend_before_sec: float,
    max_extend_after_sec: float,
) -> tuple[list[TimeSpan], list[int]]:
    matched_indexes: list[int] = []
    for index, block in enumerate(blocks):
        text = " ".join(block.lines).lower()
        if any(keyword in text for keyword in keywords):
            matched_indexes.append(index)
    if not matched_indexes:
        return [], []

    grouped_indexes: list[list[int]] = [[matched_indexes[0]]]
    for index in matched_indexes[1:]:
        previous = grouped_indexes[-1][-1]
        if blocks[index].start - blocks[previous].end <= group_gap_sec:
            grouped_indexes[-1].append(index)
        else:
            grouped_indexes.append([index])

    spans: list[TimeSpan] = []
    for group in grouped_indexes:
        start_index = group[0]
        end_index = group[-1]
        group_start_time = blocks[group[0]].start
        group_end_time = blocks[group[-1]].end

        while start_index > 0:
            gap = blocks[start_index].start - blocks[start_index - 1].end
            candidate = start_index - 1
            if gap > scene_gap_sec or group_start_time - blocks[candidate].start > max_extend_before_sec:
                break
            start_index = candidate

        while end_index + 1 < len(blocks):
            gap = blocks[end_index + 1].start - blocks[end_index].end
            candidate = end_index + 1
            if gap > scene_gap_sec or blocks[candidate].end - group_end_time > max_extend_after_sec:
                break
            end_index = candidate

        spans.append(
            TimeSpan(
                start=max(0.0, blocks[start_index].start - margin_before_sec),
                end=max(blocks[end_index].end, blocks[end_index].end + margin_after_sec),
            )
        )

    return spans, matched_indexes


def _keep_spans(duration_sec: float, cut_spans: list[TimeSpan]) -> list[TimeSpan]:
    keep: list[TimeSpan] = []
    cursor = 0.0
    for span in cut_spans:
        if span.start > cursor:
            keep.append(TimeSpan(start=cursor, end=span.start))
        cursor = max(cursor, span.end)
    if cursor < duration_sec:
        keep.append(TimeSpan(start=cursor, end=duration_sec))
    return [span for span in keep if span.end - span.start > 0.05]


def _ffprobe_duration(video: Path) -> float:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video.name}: {process.stdout}")
    return float(process.stdout.strip())


def _rewrite_srt(blocks: list[SRTBlock], cut_spans: list[TimeSpan]) -> list[SRTBlock]:
    rewritten: list[SRTBlock] = []
    for block in blocks:
        if any(_overlaps(block, span) for span in cut_spans):
            continue
        start_shift = _removed_before(block.start, cut_spans)
        end_shift = _removed_before(block.end, cut_spans)
        rewritten.append(
            SRTBlock(
                start=max(0.0, block.start - start_shift),
                end=max(0.0, block.end - end_shift),
                lines=block.lines,
            )
        )
    return rewritten


def _cut_video(
    input_video: Path,
    output_video: Path,
    keep_spans: list[TimeSpan],
    preset: str,
    crf: str,
) -> None:
    if not keep_spans:
        raise RuntimeError("No keep spans remain after explicit-content cuts.")

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for index, span in enumerate(keep_spans):
        filter_parts.append(
            f"[0:v]trim=start={span.start:.3f}:end={span.end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={span.start:.3f}:end={span.end:.3f},asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filter_parts.append(
        "".join(concat_inputs) + f"concat=n={len(keep_spans)}:v=1:a=1[vout][aout]"
    )
    process = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_video),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed for {input_video.name}: {process.stdout}")


def _clear_stale_outputs(video: Path) -> list[str]:
    removed: list[str] = []
    spans_path = video.with_suffix(".speech_spans.json")
    if spans_path.exists():
        spans_path.unlink()
        removed.append(str(spans_path))
    derived_dir = video.parent / video.stem
    if derived_dir.exists():
        shutil.rmtree(derived_dir)
        removed.append(str(derived_dir))
    return removed


def _backup_inputs(video: Path, sidecar: Path) -> Path:
    backup_dir = video.parent / f"{video.stem}.explicit_cut_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, backup_dir / video.name)
    shutil.copy2(sidecar, backup_dir / sidecar.name)
    return backup_dir


def _report_path_for(video: Path) -> Path:
    return video.parent / f"{video.stem}.explicit_cut_report.json"


def explicit_cut_report_path(video: Path) -> Path:
    return _report_path_for(video)


def _write_report(report_path: Path, report: ExplicitCutPlan | dict[str, object]) -> None:
    payload = report.to_dict() if isinstance(report, ExplicitCutPlan) else report
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_explicit_cut_plan_report(video: Path, plan: ExplicitCutPlan) -> Path:
    report_path = _report_path_for(video)
    _write_report(report_path, plan)
    return report_path


def _extract_frame(video: Path, timestamp_sec: float, output_path: Path) -> None:
    process = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, timestamp_sec):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed for {video.name}: {process.stdout}")


def _sample_frame_timestamps(span: TimeSpan, interval_sec: float, max_samples: int) -> list[float]:
    duration = max(0.0, span.end - span.start)
    if duration <= 0.0:
        return []

    interval = max(0.25, interval_sec)
    timestamps: list[float] = []
    cursor = span.start + min(interval / 2.0, duration / 2.0)
    while cursor < span.end and len(timestamps) < max_samples:
        timestamps.append(cursor)
        cursor += interval
    if not timestamps:
        timestamps.append(span.start + duration / 2.0)
    return timestamps[:max_samples]


def _score_frames_with_opennsfw2(frame_paths: list[Path]) -> list[float]:
    try:
        import opennsfw2  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "OpenNSFW2 backend requested but opennsfw2 is not installed."
        ) from error

    scores = opennsfw2.predict_images([str(path) for path in frame_paths])
    return [float(score) for score in scores]


def _score_frames_with_nudenet(frame_paths: list[Path]) -> list[float]:
    try:
        from nudenet import NudeDetector  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "NudeNet backend requested but nudenet is not installed."
        ) from error

    detector = NudeDetector()
    scores: list[float] = []
    positive_labels = {
        "EXPOSED_BREAST_F",
        "EXPOSED_GENITALIA_F",
        "EXPOSED_GENITALIA_M",
        "EXPOSED_BUTTOCKS",
        "COVERED_BREAST_F",
    }
    for frame_path in frame_paths:
        detections = detector.detect(str(frame_path))
        if not detections:
            scores.append(0.0)
            continue
        positive_scores = [
            float(item.get("score", 0.0))
            for item in detections
            if str(item.get("class", "")).upper() in positive_labels
        ]
        score = max(positive_scores, default=0.0)
        scores.append(score)
    return scores


def _verify_plan_frames(
    video: Path,
    cut_spans: list[TimeSpan],
    config: ExplicitCutConfig,
) -> tuple[bool | None, str | None, list[dict[str, object]]]:
    if not cut_spans:
        return None, None, []
    backend = config.frame_verification_backend.lower().strip()
    if backend in {"", "off", "none", "disabled"}:
        return None, None, []

    frame_paths: list[Path] = []
    timestamp_rows: list[tuple[Path, float, int]] = []
    with tempfile.TemporaryDirectory(prefix="explicit_cut_verify_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        for span_index, span in enumerate(cut_spans):
            timestamps = _sample_frame_timestamps(
                span,
                interval_sec=config.frame_interval_sec,
                max_samples=config.frame_max_samples_per_span,
            )
            for sample_index, timestamp in enumerate(timestamps):
                frame_path = temp_dir / f"span{span_index:02d}_sample{sample_index:02d}.jpg"
                _extract_frame(video, timestamp, frame_path)
                frame_paths.append(frame_path)
                timestamp_rows.append((frame_path, timestamp, span_index))

        if not frame_paths:
            return False, f"{backend}: no frames sampled", []

        if backend == "opennsfw2":
            scores = _score_frames_with_opennsfw2(frame_paths)
        elif backend == "nudenet":
            scores = _score_frames_with_nudenet(frame_paths)
        else:
            raise RuntimeError(
                "Unsupported explicit-cut frame backend: "
                f"{config.frame_verification_backend}"
            )

    samples: list[dict[str, object]] = []
    positive_hits = 0
    for score, (_, timestamp, span_index) in zip(scores, timestamp_rows):
        is_positive = score >= config.frame_nsfw_threshold
        if is_positive:
            positive_hits += 1
        samples.append(
            {
                "span_index": span_index,
                "timestamp_sec": round(timestamp, 3),
                "score": round(score, 4),
                "above_threshold": is_positive,
            }
        )

    sample_count = len(samples)
    positive_ratio = positive_hits / sample_count if sample_count else 0.0
    passed = positive_ratio >= config.frame_min_positive_ratio
    summary = (
        f"{backend}: {positive_hits}/{sample_count} sampled frame(s) "
        f">= {config.frame_nsfw_threshold:.2f} "
        f"({positive_ratio:.0%}, required {config.frame_min_positive_ratio:.0%})"
    )
    return passed, summary, samples


def build_explicit_cut_plan(video: Path, config: ExplicitCutConfig) -> ExplicitCutPlan:
    sidecar = video.with_suffix(".srt")
    if not sidecar.exists():
        raise FileNotFoundError(f"Sidecar SRT not found for {video.name}")

    content, used_encoding = _read_text_with_fallbacks(sidecar, config.sidecar_encoding)
    blocks = _parse_srt(content)
    if not blocks:
        raise ValueError(f"Sidecar SRT has no subtitle blocks for {video.name}")

    duration_sec = _ffprobe_duration(video)
    cut_spans, matched_indexes = _detect_explicit_spans(
        blocks=blocks,
        keywords=config.keywords,
        margin_before_sec=config.margin_before_sec,
        margin_after_sec=config.margin_after_sec,
        group_gap_sec=config.group_gap_sec,
        scene_gap_sec=config.scene_gap_sec,
        max_extend_before_sec=config.max_extend_before_sec,
        max_extend_after_sec=config.max_extend_after_sec,
    )
    merged_cut_spans = _merge_spans(cut_spans, gap_sec=0.05)
    cut_duration_sec = sum(span.end - span.start for span in merged_cut_spans)
    verification_passed, verification_summary, verification_samples = _verify_plan_frames(
        video=video,
        cut_spans=merged_cut_spans,
        config=config,
    )
    return ExplicitCutPlan(
        video_file=video.name,
        sidecar_file=sidecar.name,
        sidecar_encoding_used=used_encoding,
        keywords=config.keywords,
        matched_block_indexes=matched_indexes,
        cut_spans=merged_cut_spans,
        original_duration_sec=duration_sec,
        cut_duration_sec=cut_duration_sec,
        result_duration_sec=max(0.0, duration_sec - cut_duration_sec),
        frame_verification_backend=config.frame_verification_backend,
        frame_verification_passed=verification_passed,
        frame_verification_summary=verification_summary,
        frame_verification_samples=verification_samples,
    )


def apply_explicit_cut_plan(
    video: Path,
    plan: ExplicitCutPlan,
    config: ExplicitCutConfig,
) -> ExplicitCutPlan:
    sidecar = video.with_suffix(".srt")
    if not sidecar.exists():
        raise FileNotFoundError(f"Sidecar SRT not found for {video.name}")
    if not plan.cut_spans:
        return ExplicitCutPlan(
            **{
                **plan.__dict__,
                "applied": False,
                "backup_dir": None,
                "removed_stale_outputs": [],
            }
        )

    content, used_encoding = _read_text_with_fallbacks(sidecar, plan.sidecar_encoding_used)
    blocks = _parse_srt(content)
    keep_spans = _keep_spans(plan.original_duration_sec, plan.cut_spans)
    rewritten_blocks = _rewrite_srt(blocks, plan.cut_spans)
    backup_path = _backup_inputs(video, sidecar)

    with tempfile.TemporaryDirectory(prefix="explicit_cut_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        temp_video = temp_dir / video.name
        temp_sidecar = temp_dir / sidecar.name
        _cut_video(video, temp_video, keep_spans, preset=config.preset, crf=config.crf)
        temp_sidecar.write_text(_serialize_srt(rewritten_blocks), encoding=used_encoding)
        shutil.move(str(temp_video), str(video))
        shutil.move(str(temp_sidecar), str(sidecar))

    removed_artifacts = _clear_stale_outputs(video)
    return ExplicitCutPlan(
        video_file=plan.video_file,
        sidecar_file=plan.sidecar_file,
        sidecar_encoding_used=used_encoding,
        keywords=plan.keywords,
        matched_block_indexes=plan.matched_block_indexes,
        cut_spans=plan.cut_spans,
        original_duration_sec=plan.original_duration_sec,
        cut_duration_sec=plan.cut_duration_sec,
        result_duration_sec=plan.result_duration_sec,
        frame_verification_backend=plan.frame_verification_backend,
        frame_verification_passed=plan.frame_verification_passed,
        frame_verification_summary=plan.frame_verification_summary,
        frame_verification_samples=plan.frame_verification_samples,
        applied=True,
        backup_dir=str(backup_path),
        removed_stale_outputs=removed_artifacts,
    )


def _process_video(
    video: Path,
    config: ExplicitCutConfig,
    analyze_only: bool,
) -> None:
    report_path = _report_path_for(video)
    backup_dir = video.parent / f"{video.stem}.explicit_cut_backup"
    if report_path.exists() and backup_dir.exists() and not config.force:
        logger.info("Skipping %s: explicit cut already applied (use force to re-run)", video.name)
        return

    try:
        plan = build_explicit_cut_plan(video, config)
    except FileNotFoundError:
        logger.warning("Skipping %s: sidecar SRT not found", video.name)
        return
    except ValueError:
        logger.warning("Skipping %s: sidecar SRT has no subtitle blocks", video.name)
        return

    _write_report(report_path, plan)
    if not plan.cut_spans:
        logger.info("No explicit-content subtitle matches found for %s", video.name)
        logger.info("  Report: %s", report_path)
        return

    logger.info(
        "Prepared %s explicit span(s) for %s (matched subtitle cues: %s)",
        len(plan.cut_spans),
        video.name,
        len(plan.matched_block_indexes),
    )
    for index, span in enumerate(plan.cut_spans, start=1):
        logger.info(
            "  Cut %d: %s -> %s (%.2fs)",
            index,
            _seconds_to_srt_time(span.start),
            _seconds_to_srt_time(span.end),
            span.end - span.start,
        )

    if plan.frame_verification_summary:
        logger.info("  Frame verification: %s", plan.frame_verification_summary)

    logger.info("  Report: %s", report_path)
    if analyze_only:
        logger.info("  Analyze-only mode: waiting for approval before applying cuts.")
        return

    applied_plan = apply_explicit_cut_plan(video, plan, config)
    _write_report(report_path, applied_plan)
    logger.info("Explicit-content cut complete for %s", video.name)
    logger.info("  Backup: %s", applied_plan.backup_dir)
    logger.info("  Report: %s", report_path)


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

    config = load_explicit_cut_config_from_env()
    analyze_only = _parse_bool(
        _first_env("TRANSCRIBE_EXPLICIT_CUT_ANALYZE_ONLY", default="false")
    )

    logger.info("Found %d video(s) for explicit-content cut", len(videos))
    for index, video in enumerate(videos, start=1):
        logger.info("")
        logger.info("=" * 60)
        logger.info("Video %d/%d: %s", index, len(videos), video.name)
        logger.info("=" * 60)
        _process_video(
            video=video,
            config=config,
            analyze_only=analyze_only,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
