"""
Transcribe video files using Whisper and process SRT files with translation.

Directory structure for each video:
video_folder/
├── video_file.mp4
├── video_file.srt (translated, in sidecar SRT encoding)
└── video_file_name/
    ├── original/
    │   └── video_file.srt (original transcription, utf-8)
    ├── stdout.txt
    ├── translated_windows1251/
    │   └── video_file.srt
    └── translated_utf8/
        └── video_file.srt
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import time
from pathlib import Path
import wave
from array import array

import nltk
import torch
from dotenv import load_dotenv
from g2p_en import G2p
from transformers import MarianMTModel, MarianTokenizer

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


@dataclass
class SpeechSpan:
    start: float
    end: float


@dataclass
class SRTBlock:
    start: float
    end: float
    lines: list[str]


def _parse_bool(value: str) -> bool:
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
    chunks = re.split(r"\r?\n\r?\n+", content.strip())

    for chunk in chunks:
        lines = chunk.splitlines()
        if len(lines) < 2:
            continue

        line_index = 0
        if lines[0].strip().isdigit():
            line_index = 1

        if line_index >= len(lines) or "-->" not in lines[line_index]:
            continue

        timecode = lines[line_index]
        start_raw, end_raw = [part.strip() for part in timecode.split("-->", maxsplit=1)]
        text_lines = [line for line in lines[line_index + 1:] if line.strip()]
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


def _merge_spans(spans: list[SpeechSpan]) -> list[SpeechSpan]:
    if not spans:
        return []

    merged: list[SpeechSpan] = []
    ordered = sorted(spans, key=lambda x: x.start)

    for span in ordered:
        if not merged or span.start > merged[-1].end:
            merged.append(SpeechSpan(start=span.start, end=span.end))
            continue

        merged[-1].end = max(merged[-1].end, span.end)

    return merged


def _spans_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


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


class SpeechSpanDetector:
    def __init__(
            self,
            margin_sec: float,
            threshold: float,
            min_speech_duration_sec: float,
            min_silence_duration_sec: float,
            sample_rate: int = 16_000,
            span_file_suffix: str = ".speech_spans.json",
    ):
        self.margin_sec = margin_sec
        self.threshold = threshold
        self.min_speech_duration_sec = min_speech_duration_sec
        self.min_silence_duration_sec = min_silence_duration_sec
        self.sample_rate = sample_rate
        self.span_file_suffix = span_file_suffix
        self._model = None
        self._get_speech_timestamps = None

    def spans_path_for(self, video_path: Path) -> Path:
        return video_path.with_suffix(self.span_file_suffix)

    def load_spans(self, spans_path: Path) -> list[SpeechSpan]:
        payload = json.loads(spans_path.read_text(encoding="utf-8"))
        loaded = [
            SpeechSpan(start=float(item["start"]), end=float(item["end"]))
            for item in payload.get("spans", [])
        ]
        return _merge_spans(loaded)

    def save_spans(self, spans_path: Path, video_path: Path, spans: list[SpeechSpan]) -> None:
        payload = {
            "video_file": video_path.name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "margin_sec": self.margin_sec,
            "sample_rate": self.sample_rate,
            "spans": [
                {"start": round(span.start, 3), "end": round(span.end, 3)} for span in spans
            ],
        }
        spans_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def detect_and_save(self, video_path: Path, overwrite: bool = False) -> tuple[Path, list[SpeechSpan]]:
        spans_path = self.spans_path_for(video_path)
        if spans_path.exists() and not overwrite:
            logger.info("  Using existing speech spans: %s", spans_path.name)
            return spans_path, self.load_spans(spans_path)

        logger.info("  Detecting speech spans for %s", video_path.name)
        start_time = time.perf_counter()
        spans = self.detect(video_path)
        duration_sec = time.perf_counter() - start_time
        self.save_spans(spans_path, video_path, spans)
        logger.info(
            "  Saved speech spans: %s (%d span(s), %.2fs)",
            spans_path.name,
            len(spans),
            duration_sec,
        )
        return spans_path, spans

    def detect(self, video_path: Path) -> list[SpeechSpan]:
        self._ensure_model_loaded()

        with tempfile.TemporaryDirectory(prefix="vad_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            self._extract_mono_audio(video_path, wav_path)

            wav = self._read_wav_mono_tensor(wav_path)
            timestamps = self._get_speech_timestamps(
                wav,
                self._model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=int(self.min_speech_duration_sec * 1000),
                min_silence_duration_ms=int(self.min_silence_duration_sec * 1000),
            )
            duration_sec = len(wav) / float(self.sample_rate)

        spans: list[SpeechSpan] = []
        for timestamp in timestamps:
            start_sec = max(0.0, timestamp["start"] / self.sample_rate - self.margin_sec)
            end_sec = min(duration_sec, timestamp["end"] / self.sample_rate + self.margin_sec)
            if end_sec > start_sec:
                spans.append(SpeechSpan(start=start_sec, end=end_sec))

        return _merge_spans(spans)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info("  Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]

    def _extract_mono_audio(self, video_path: Path, wav_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            str(wav_path),
        ]
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Failed to extract audio for VAD from {video_path.name}: {process.stdout}"
            )

    def _read_wav_mono_tensor(self, wav_path: Path) -> torch.Tensor:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Unexpected sample rate in {wav_path.name}: {sample_rate} "
                f"(expected {self.sample_rate})"
            )
        if channels != 1:
            raise RuntimeError(f"Expected mono WAV for VAD, got {channels} channel(s)")
        if sample_width != 2:
            raise RuntimeError(
                f"Expected 16-bit PCM WAV for VAD, got sample width {sample_width}"
            )

        pcm = array("h")
        pcm.frombytes(frames)
        waveform = torch.tensor(pcm, dtype=torch.float32) / 32768.0
        return waveform


def resolve_translation_model(source_language: str, target_language: str) -> str:
    language_pair = f"{source_language.lower()}-{target_language.lower()}"

    predefined_models = {
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en",
        "fr-ru": "Helsinki-NLP/opus-mt-fr-ru",
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    }

    model_name = predefined_models.get(language_pair)

    if model_name:
        return model_name

    raise ValueError(
        f"Unsupported translation pair '{language_pair}'. "
        "Set TRANSCRIBE_TRANSLATION_MODEL explicitly, "
        "for example: Helsinki-NLP/opus-mt-fr-en"
    )


class TranslationModel:
    """Responsible for translation using MarianMT model."""

    def __init__(
            self,
            model_name: str,
            hf_token: str | None = None,
            local_files_only: bool = False,
    ):
        auth_kwargs = {}
        if hf_token:
            auth_kwargs["token"] = hf_token

        self.tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **auth_kwargs,
        )

        self.model = MarianMTModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **auth_kwargs,
        )

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def translate(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class EnglishRespeller:
    """Responsible for phoneme to respelling conversion."""

    VOWELS = {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }

    PHONEME_MAP = {
        "AA": "A",
        "AE": "A",
        "AH": "U",
        "AO": "AW",
        "AW": "OW",
        "AY": "AY",
        "EH": "E",
        "ER": "ER",
        "EY": "AY",
        "IH": "I",
        "IY": "EE",
        "OW": "OH",
        "OY": "OY",
        "UH": "U",
        "UW": "OO",
        "B": "B",
        "CH": "CH",
        "D": "D",
        "DH": "TH",
        "F": "F",
        "G": "G",
        "HH": "H",
        "JH": "J",
        "K": "K",
        "L": "L",
        "M": "M",
        "N": "N",
        "NG": "NG",
        "P": "P",
        "R": "R",
        "S": "S",
        "SH": "SH",
        "T": "T",
        "TH": "TH",
        "V": "V",
        "W": "W",
        "Y": "Y",
        "Z": "Z",
        "ZH": "ZH",
    }

    def __init__(self):
        self.g2p = G2p()

    def respell(self, text: str) -> str:
        phonemes = self.g2p(text)

        words = []
        current = []

        for p in phonemes:
            if p == " ":
                if current:
                    words.append(self._render_word(current))
                    current = []
                continue

            current.append(p)

        if current:
            words.append(self._render_word(current))

        return " ".join(words)

    def _render_word(self, phonemes: list[str]) -> str:
        syllables = []
        current = []

        for p in phonemes:
            base = re.sub(r"\d", "", p)

            if base in self.VOWELS and current:
                syllables.append(current)
                current = []

            current.append(p)

        if current:
            syllables.append(current)

        rendered = []

        for syll in syllables:
            stress = any(p.endswith("1") for p in syll)

            parts = []

            for p in syll:
                base = re.sub(r"\d", "", p)
                letters = self.PHONEME_MAP.get(base, base)

                if stress:
                    letters = letters.upper()
                else:
                    letters = letters.lower()

                parts.append(letters)

            rendered.append("".join(parts))

        return "-".join(rendered)


class SRTTranslator:
    """Responsible for SRT parsing and translation."""

    TIMESTAMP = re.compile(r"\d\d:\d\d:\d\d,\d\d\d")

    def __init__(
            self,
            translator: TranslationModel,
            respeller: EnglishRespeller | None = None,
            batch_size: int = 32,
    ):
        self.translator = translator
        self.respeller = respeller
        self.batch_size = batch_size

    def translate_file(
            self,
            input_path: Path,
            output_path: Path,
            append: bool = True,
    ) -> None:
        lines = input_path.read_text(encoding="utf-8").splitlines()

        text_lines = []
        indices = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if (
                    stripped.isdigit()
                    or "-->" in line
                    or not stripped
                    or self.TIMESTAMP.search(line)
            ):
                continue

            text_lines.append(stripped)
            indices.append(i)

        translations = self._translate_batches(text_lines)

        mapping = dict(zip(indices, translations))

        out = []

        for i, line in enumerate(lines):
            if i not in mapping:
                out.append(line)
                continue

            translated = mapping[i]

            if append:
                out.append(line)

                if self.respeller:
                    out.append(self.respeller.respell(line))

                out.append(translated)
            else:
                out.append(translated)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text("\n".join(out), encoding="utf-8")

    def _translate_batches(self, lines: list[str]) -> list[str]:
        results = []

        for i in range(0, len(lines), self.batch_size):
            batch = lines[i: i + self.batch_size]
            results.extend(self.translator.translate(batch))

        return results


class WhisperTranscriber:
    """Responsible for Whisper CLI transcription."""

    def __init__(self, language: str):
        self.language = language

    def transcribe(
            self,
            video_path: Path,
            output_dir: Path,
            speech_spans: list[SpeechSpan] | None = None,
    ) -> tuple[Path, str]:
        video_name = video_path.stem

        # Create original subfolder
        original_dir = output_dir / "original"
        original_dir.mkdir(exist_ok=True)

        srt_path = original_dir / f"{video_name}.srt"
        stdout_path = output_dir / "stdout.txt"

        if srt_path.exists() and stdout_path.exists():
            logger.info("Transcription already exists")
            return srt_path, stdout_path.read_text(encoding="utf-8")

        if speech_spans:
            logger.info("Starting span-aware Whisper transcription...")
            stdout_content = self._transcribe_with_spans(
                video_path=video_path,
                srt_path=srt_path,
                stdout_path=stdout_path,
                speech_spans=speech_spans,
            )
            return srt_path, stdout_content

        cmd = [
            "whisper",
            str(video_path),
            "--language",
            self.language,
            "--output_format",
            "srt",
        ]

        logger.info("Starting Whisper transcription...")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=video_path.parent,
            bufsize=1,
            env=env,
        )

        lines = []

        if process.stdout:
            for line in process.stdout:
                line = line.rstrip()
                lines.append(line)
                logger.info("Whisper: %s", line)

        process.wait()

        stdout_content = "\n".join(lines)
        stdout_path.write_text(stdout_content, encoding="utf-8")

        whisper_srt = video_path.with_suffix(".srt")

        if whisper_srt.exists():
            whisper_srt.rename(srt_path)

        if process.returncode != 0:
            raise RuntimeError(
                f"Whisper failed with return code {process.returncode} for {video_path.name}"
            )

        if not srt_path.exists():
            raise RuntimeError(
                f"Whisper finished without creating SRT for {video_path.name}. "
                "Check stdout.txt for details."
            )

        return srt_path, stdout_content

    def _transcribe_with_spans(
            self,
            video_path: Path,
            srt_path: Path,
            stdout_path: Path,
            speech_spans: list[SpeechSpan],
    ) -> str:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        all_segments: list[SRTBlock] = []
        stdout_lines: list[str] = []

        with tempfile.TemporaryDirectory(prefix="whisper_spans_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)

            for index, span in enumerate(speech_spans, start=1):
                span_wav = temp_dir / f"span_{index:04}.wav"
                span_json = temp_dir / f"span_{index:04}.json"

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{span.start:.3f}",
                    "-to",
                    f"{span.end:.3f}",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(span_wav),
                ]
                ffmpeg_process = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                    cwd=video_path.parent,
                )
                stdout_lines.append(
                    f"[span {index}] ffmpeg return code: {ffmpeg_process.returncode}"
                )
                if ffmpeg_process.stdout:
                    stdout_lines.append(ffmpeg_process.stdout.rstrip())
                if ffmpeg_process.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg failed for span {index} in {video_path.name}"
                    )

                whisper_cmd = [
                    "whisper",
                    str(span_wav),
                    "--language",
                    self.language,
                    "--output_format",
                    "json",
                    "--output_dir",
                    str(temp_dir),
                ]
                whisper_process = subprocess.run(
                    whisper_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                    cwd=video_path.parent,
                )
                stdout_lines.append(
                    f"[span {index}] whisper return code: {whisper_process.returncode}"
                )
                if whisper_process.stdout:
                    stdout_lines.append(whisper_process.stdout.rstrip())
                if whisper_process.returncode != 0:
                    raise RuntimeError(
                        f"Whisper failed for span {index} in {video_path.name}"
                    )

                if not span_json.exists():
                    raise RuntimeError(
                        f"Whisper did not generate JSON for span {index} in {video_path.name}"
                    )

                payload = json.loads(span_json.read_text(encoding="utf-8"))
                segments = payload.get("segments", [])

                for segment in segments:
                    text = str(segment.get("text", "")).strip()
                    start = float(segment.get("start", 0.0)) + span.start
                    end = float(segment.get("end", 0.0)) + span.start
                    if text and end > start:
                        all_segments.append(SRTBlock(start=start, end=end, lines=[text]))

        all_segments = sorted(all_segments, key=lambda block: block.start)
        srt_path.write_text(_serialize_srt(all_segments), encoding="utf-8")

        stdout_content = "\n".join(stdout_lines)
        stdout_path.write_text(stdout_content, encoding="utf-8")

        return stdout_content


class VideoPipeline:
    """Responsible for orchestrating the video processing workflow."""

    def __init__(
            self,
            video_folder: Path,
            translator: SRTTranslator | None,
            transcriber: WhisperTranscriber,
            speech_span_detector: SpeechSpanDetector | None = None,
            use_speech_spans_for_whisper: bool = True,
            filter_sidecar_srt_by_speech_spans: bool = True,
            detect_speech_spans_if_missing: bool = True,
            overwrite_speech_spans: bool = False,
            single_video_name: str | None = None,
            speech_spans_only_mode: bool = False,
            speech_filter_rescue_dense_runs: bool = True,
            speech_filter_rescue_min_cues: int = 4,
            speech_filter_rescue_min_duration_sec: float = 8.0,
            speech_filter_rescue_max_duration_sec: float = 90.0,
            speech_filter_rescue_min_density: float = 0.08,
            speech_filter_rescue_tiny_runs: bool = True,
            speech_filter_rescue_tiny_max_cues: int = 3,
            speech_filter_rescue_tiny_max_duration_sec: float = 4.0,
            speech_filter_rescue_tiny_neighbor_gap_sec: float = 8.0,
            sidecar_srt_encoding: str = "utf-8",
            output_folder: Path | None = None,
    ):
        self.video_folder = video_folder
        self.translator = translator
        self.transcriber = transcriber
        self.speech_span_detector = speech_span_detector
        self.use_speech_spans_for_whisper = use_speech_spans_for_whisper
        self.filter_sidecar_srt_by_speech_spans = filter_sidecar_srt_by_speech_spans
        self.detect_speech_spans_if_missing = detect_speech_spans_if_missing
        self.overwrite_speech_spans = overwrite_speech_spans
        self.single_video_name = single_video_name
        self.speech_spans_only_mode = speech_spans_only_mode
        self.speech_filter_rescue_dense_runs = speech_filter_rescue_dense_runs
        self.speech_filter_rescue_min_cues = speech_filter_rescue_min_cues
        self.speech_filter_rescue_min_duration_sec = speech_filter_rescue_min_duration_sec
        self.speech_filter_rescue_max_duration_sec = speech_filter_rescue_max_duration_sec
        self.speech_filter_rescue_min_density = speech_filter_rescue_min_density
        self.speech_filter_rescue_tiny_runs = speech_filter_rescue_tiny_runs
        self.speech_filter_rescue_tiny_max_cues = speech_filter_rescue_tiny_max_cues
        self.speech_filter_rescue_tiny_max_duration_sec = (
            speech_filter_rescue_tiny_max_duration_sec
        )
        self.speech_filter_rescue_tiny_neighbor_gap_sec = (
            speech_filter_rescue_tiny_neighbor_gap_sec
        )
        self.sidecar_srt_encoding = sidecar_srt_encoding
        self.output_folder = output_folder

    def run(self) -> None:
        if not self.video_folder.exists():
            logger.error("Video folder does not exist: %s", self.video_folder)
            return

        if self.output_folder:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info("Output folder: %s", self.output_folder)

        videos = list(self.video_folder.glob("*.mp4"))
        if self.single_video_name:
            target = self.single_video_name.lower().strip()
            videos = [
                video
                for video in videos
                if video.name.lower() == target or video.stem.lower() == target
            ]

        if not videos:
            logger.warning("No .mp4 files found in %s", self.video_folder)
            return

        logger.info("Found %d video(s) to process", len(videos))

        for idx, video in enumerate(videos, 1):
            logger.info("")
            logger.info("=" * 60)
            logger.info("Video %d/%d: %s", idx, len(videos), video.name)
            logger.info("=" * 60)

            try:
                self._process_video(video)
            except Exception as error:
                logger.exception("Failed to process %s: %s", video.name, error)
                continue

    def _process_video(self, video: Path) -> None:
        name = video.stem
        output_dir = video.parent / name

        output_dir.mkdir(exist_ok=True)

        original_srt = output_dir / "original" / f"{name}.srt"
        stdout_path = output_dir / "stdout.txt"
        translated_windows1251 = output_dir / "translated_windows1251" / f"{name}.srt"
        translated_utf8 = output_dir / "translated_utf8" / f"{name}.srt"
        duplicate_path = video.parent / f"{name}.srt"

        speech_spans = self._resolve_speech_spans(video)
        if speech_spans and self.filter_sidecar_srt_by_speech_spans:
            self._filter_sidecar_srt(video, speech_spans)

        if self.speech_spans_only_mode:
            logger.info("  Speech spans only mode: skipping Whisper/translation/duplicate")
            return

        transcription_exists = original_srt.exists() and stdout_path.exists()
        translation_exists = translated_windows1251.exists() and translated_utf8.exists()
        duplicate_exists = duplicate_path.exists()

        if transcription_exists and translation_exists and duplicate_exists:
            logger.info("Skipping %s: all outputs already exist", video.name)

            if self.output_folder:
                self._move_to_output_folder(video, output_dir)

            return

        logger.info("Processing: %s", video.name)

        if transcription_exists:
            logger.info("  Transcription already exists, skipping Whisper")
            srt_path = original_srt
        else:
            srt_path, _ = self.transcriber.transcribe(
                video,
                output_dir,
                speech_spans=speech_spans if self.use_speech_spans_for_whisper else None,
            )
            logger.info("  Created SRT: %s", srt_path)

        if translation_exists:
            logger.info("  Translation already exists, skipping")
        else:
            if self.translator is None:
                logger.info("  Translator is disabled, skipping translation")
                return

            self.translator.translate_file(srt_path, translated_utf8, append=True)

            translated_content = translated_utf8.read_text(encoding="utf-8")
            translated_windows1251.parent.mkdir(parents=True, exist_ok=True)
            translated_windows1251.write_text(
                translated_content,
                encoding="windows-1251",
                errors="replace",
            )

            logger.info("  Created translated files:")
            logger.info("    %s", translated_windows1251)
            logger.info("    %s", translated_utf8)

        if duplicate_exists:
            logger.info("  Duplicate already exists, skipping")
        else:
            self._duplicate(video, translated_utf8)
            logger.info(
                "  Created duplicate: %s (encoding: %s)",
                duplicate_path,
                self.sidecar_srt_encoding,
            )

        logger.info("  Done!")

        if self.output_folder:
            self._move_to_output_folder(video, output_dir)

    def _duplicate(self, video: Path, srt: Path) -> None:
        content = srt.read_text(encoding="utf-8")
        duplicate = video.parent / f"{video.stem}.srt"
        duplicate.write_text(
            content,
            encoding=self.sidecar_srt_encoding,
            errors="replace",
        )

    def _resolve_speech_spans(self, video: Path) -> list[SpeechSpan] | None:
        if not self.speech_span_detector:
            return None

        spans_path = self.speech_span_detector.spans_path_for(video)

        if spans_path.exists() and not self.overwrite_speech_spans:
            logger.info("  Loading speech spans: %s", spans_path.name)
            return self.speech_span_detector.load_spans(spans_path)

        if not self.detect_speech_spans_if_missing:
            return None

        _, spans = self.speech_span_detector.detect_and_save(
            video_path=video,
            overwrite=self.overwrite_speech_spans,
        )
        return spans

    def _filter_sidecar_srt(self, video: Path, speech_spans: list[SpeechSpan]) -> None:
        sidecar_srt = video.with_suffix(".srt")
        if not sidecar_srt.exists():
            return

        content, used_encoding = _read_text_with_fallbacks(
            sidecar_srt,
            self.sidecar_srt_encoding,
        )
        blocks = _parse_srt(content)
        if not blocks:
            return

        keep_flags = [
            any(
                _spans_overlap(block.start, block.end, span.start, span.end)
                for span in speech_spans
            )
            for block in blocks
        ]

        if self.speech_filter_rescue_dense_runs:
            self._rescue_dense_middle_runs(blocks, keep_flags)
        if self.speech_filter_rescue_tiny_runs:
            self._rescue_tiny_middle_runs(blocks, keep_flags)

        filtered = [block for block, keep in zip(blocks, keep_flags, strict=False) if keep]

        if len(filtered) == len(blocks):
            logger.info("  Sidecar SRT already aligned with speech spans")
            return

        sidecar_srt.write_text(_serialize_srt(filtered), encoding=used_encoding)
        logger.info(
            "  Filtered sidecar SRT by speech spans: %s (%d -> %d cues)",
            sidecar_srt.name,
            len(blocks),
            len(filtered),
        )

    def _rescue_dense_middle_runs(
            self,
            blocks: list[SRTBlock],
            keep_flags: list[bool],
    ) -> None:
        index = 0
        while index < len(blocks):
            if keep_flags[index]:
                index += 1
                continue

            run_start = index
            while index + 1 < len(blocks) and not keep_flags[index + 1]:
                index += 1
            run_end = index

            has_prev_kept = run_start > 0 and keep_flags[run_start - 1]
            has_next_kept = run_end + 1 < len(blocks) and keep_flags[run_end + 1]
            if not (has_prev_kept and has_next_kept):
                index += 1
                continue

            run_cues = run_end - run_start + 1
            run_duration = blocks[run_end].end - blocks[run_start].start
            density = run_cues / run_duration if run_duration > 0 else float("inf")

            should_rescue = (
                run_cues >= self.speech_filter_rescue_min_cues
                and self.speech_filter_rescue_min_duration_sec
                <= run_duration
                <= self.speech_filter_rescue_max_duration_sec
                and density >= self.speech_filter_rescue_min_density
            )
            if should_rescue:
                for i in range(run_start, run_end + 1):
                    keep_flags[i] = True
                logger.info(
                    "  Rescued subtitle run in middle: %.3fs..%.3fs (%d cues)",
                    blocks[run_start].start,
                    blocks[run_end].end,
                    run_cues,
                )

            index += 1

    def _rescue_tiny_middle_runs(
            self,
            blocks: list[SRTBlock],
            keep_flags: list[bool],
    ) -> None:
        index = 0
        while index < len(blocks):
            if keep_flags[index]:
                index += 1
                continue

            run_start = index
            while index + 1 < len(blocks) and not keep_flags[index + 1]:
                index += 1
            run_end = index

            has_prev_kept = run_start > 0 and keep_flags[run_start - 1]
            has_next_kept = run_end + 1 < len(blocks) and keep_flags[run_end + 1]
            if not (has_prev_kept and has_next_kept):
                index += 1
                continue

            run_cues = run_end - run_start + 1
            run_duration = blocks[run_end].end - blocks[run_start].start
            gap_from_prev = blocks[run_start].start - blocks[run_start - 1].end
            gap_to_next = blocks[run_end + 1].start - blocks[run_end].end

            should_rescue = (
                run_cues <= self.speech_filter_rescue_tiny_max_cues
                and run_duration <= self.speech_filter_rescue_tiny_max_duration_sec
                and gap_from_prev <= self.speech_filter_rescue_tiny_neighbor_gap_sec
                and gap_to_next <= self.speech_filter_rescue_tiny_neighbor_gap_sec
            )
            if should_rescue:
                for i in range(run_start, run_end + 1):
                    keep_flags[i] = True
                logger.info(
                    "  Rescued tiny subtitle run in middle: %.3fs..%.3fs (%d cues)",
                    blocks[run_start].start,
                    blocks[run_end].end,
                    run_cues,
                )

            index += 1

    def _move_to_output_folder(self, video: Path, output_dir: Path) -> None:
        video_name = video.stem
        dest_dir = self.output_folder / video_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_video = dest_dir / f"{video_name}.mp4"

        if video.exists() and not dest_video.exists():
            shutil.move(str(video), str(dest_video))
            logger.info("  Moved video to: %s", dest_video)

        if output_dir.exists():
            for item in output_dir.iterdir():
                dest_item = dest_dir / item.name
                if not dest_item.exists():
                    shutil.move(str(item), str(dest_item))

            if not any(output_dir.iterdir()):
                output_dir.rmdir()

            logger.info("  Moved output directory to: %s", dest_dir)


def main() -> None:
    video_folder_raw = _first_env("TRANSCRIBE_VIDEO_FOLDER", "VIDEO_FOLDER")
    if not video_folder_raw:
        raise RuntimeError(
            "TRANSCRIBE_VIDEO_FOLDER (or legacy VIDEO_FOLDER) is not set in .env"
        )

    video_folder = Path(video_folder_raw)

    sidecar_srt_encoding = _first_env(
        "TRANSCRIBE_SIDECAR_SRT_ENCODING",
        "TRANSCRIBE_DUPLICATE_SRT_ENCODING",
        "DUPLICATE_SRT_ENCODING",
        default="utf-8",
    )
    speech_spans_enabled_raw = _first_env(
        "TRANSCRIBE_ENABLE_SPEECH_SPANS",
        default="true",
    )
    speech_spans_detect_if_missing_raw = _first_env(
        "TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING",
        default="true",
    )
    speech_spans_overwrite_raw = _first_env(
        "TRANSCRIBE_SPEECH_SPANS_OVERWRITE",
        default="false",
    )
    speech_spans_use_for_whisper_raw = _first_env(
        "TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER",
        default="true",
    )
    speech_spans_filter_sidecar_raw = _first_env(
        "TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS",
        default="true",
    )
    speech_spans_only_mode_raw = _first_env(
        "TRANSCRIBE_SPEECH_SPANS_ONLY_MODE",
        default="false",
    )
    speech_spans_margin_sec_raw = _first_env(
        "TRANSCRIBE_SPEECH_SPAN_MARGIN_SEC",
        default="0.35",
    )
    speech_vad_threshold_raw = _first_env(
        "TRANSCRIBE_SPEECH_VAD_THRESHOLD",
        default="0.3",
    )
    speech_min_duration_raw = _first_env(
        "TRANSCRIBE_SPEECH_MIN_DURATION_SEC",
        default="0.15",
    )
    speech_min_silence_raw = _first_env(
        "TRANSCRIBE_SPEECH_MIN_SILENCE_SEC",
        default="0.2",
    )
    speech_span_file_suffix = _first_env(
        "TRANSCRIBE_SPEECH_SPAN_FILE_SUFFIX",
        default=".speech_spans.json",
    )
    speech_filter_rescue_dense_runs_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_DENSE_RUNS",
        default="true",
    )
    speech_filter_rescue_min_cues_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_MIN_CUES",
        default="4",
    )
    speech_filter_rescue_min_duration_sec_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_MIN_DURATION_SEC",
        default="8",
    )
    speech_filter_rescue_max_duration_sec_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_MAX_DURATION_SEC",
        default="90",
    )
    speech_filter_rescue_min_density_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_MIN_DENSITY",
        default="0.08",
    )
    speech_filter_rescue_tiny_runs_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_RUNS",
        default="true",
    )
    speech_filter_rescue_tiny_max_cues_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_MAX_CUES",
        default="3",
    )
    speech_filter_rescue_tiny_max_duration_sec_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_MAX_DURATION_SEC",
        default="4",
    )
    speech_filter_rescue_tiny_neighbor_gap_sec_raw = _first_env(
        "TRANSCRIBE_SPEECH_FILTER_RESCUE_TINY_NEIGHBOR_GAP_SEC",
        default="8",
    )
    single_video_name = _first_env("TRANSCRIBE_SINGLE_VIDEO", default=None)

    whisper_language = _first_env(
        "TRANSCRIBE_WHISPER_LANGUAGE",
        "LANGUAGE",
        default="en",
    )
    translation_source_language = _first_env(
        "TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE",
        default=whisper_language,
    )
    translation_target_language = _first_env(
        "TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE",
        default="ru",
    )
    translation_model_name = _first_env(
        "TRANSCRIBE_TRANSLATION_MODEL",
        default=None,
    )
    hf_token = _first_env("TRANSCRIBE_HF_TOKEN", "HF_TOKEN", default=None)
    output_folder = _first_env("TRANSCRIBE_OUTPUT_FOLDER", "OUTPUT_FOLDER", default=None)
    phonetic_enabled_raw = _first_env("TRANSCRIBE_ENABLE_PHONETIC", default="true")
    phonetic_language = _first_env("TRANSCRIBE_PHONETIC_LANGUAGE", default="en")
    offline_mode_raw = _first_env("TRANSCRIBE_OFFLINE_MODE", default="false")

    if translation_model_name is None:
        translation_model_name = resolve_translation_model(
            translation_source_language,
            translation_target_language,
        )

    if output_folder:
        output_folder = Path(output_folder)

    speech_spans_enabled = _parse_bool(speech_spans_enabled_raw)
    speech_spans_detect_if_missing = _parse_bool(speech_spans_detect_if_missing_raw)
    speech_spans_overwrite = _parse_bool(speech_spans_overwrite_raw)
    speech_spans_use_for_whisper = _parse_bool(speech_spans_use_for_whisper_raw)
    speech_spans_filter_sidecar = _parse_bool(speech_spans_filter_sidecar_raw)
    speech_spans_only_mode = _parse_bool(speech_spans_only_mode_raw)
    speech_spans_margin_sec = float(speech_spans_margin_sec_raw)
    speech_vad_threshold = float(speech_vad_threshold_raw)
    speech_min_duration_sec = float(speech_min_duration_raw)
    speech_min_silence_sec = float(speech_min_silence_raw)
    speech_filter_rescue_dense_runs = _parse_bool(speech_filter_rescue_dense_runs_raw)
    speech_filter_rescue_min_cues = int(speech_filter_rescue_min_cues_raw)
    speech_filter_rescue_min_duration_sec = float(speech_filter_rescue_min_duration_sec_raw)
    speech_filter_rescue_max_duration_sec = float(speech_filter_rescue_max_duration_sec_raw)
    speech_filter_rescue_min_density = float(speech_filter_rescue_min_density_raw)
    speech_filter_rescue_tiny_runs = _parse_bool(speech_filter_rescue_tiny_runs_raw)
    speech_filter_rescue_tiny_max_cues = int(speech_filter_rescue_tiny_max_cues_raw)
    speech_filter_rescue_tiny_max_duration_sec = float(
        speech_filter_rescue_tiny_max_duration_sec_raw
    )
    speech_filter_rescue_tiny_neighbor_gap_sec = float(
        speech_filter_rescue_tiny_neighbor_gap_sec_raw
    )

    if speech_spans_only_mode:
        output_folder = None

    offline_mode = offline_mode_raw.lower() in {"1", "true", "yes", "on"}
    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    model = None
    if not speech_spans_only_mode:
        try:
            model = TranslationModel(
                translation_model_name,
                hf_token=hf_token,
                local_files_only=offline_mode,
            )
        except ImportError as error:
            raise RuntimeError(
                "Missing dependency for Marian translation model. "
                "Install dependencies with 'uv sync' (protobuf/sentencepiece are required)."
            ) from error

    phonetic_enabled = phonetic_enabled_raw.lower() in {"1", "true", "yes", "on"}
    use_english_respeller = (
        phonetic_enabled
        and phonetic_language.lower().startswith("en")
        and translation_source_language.lower().startswith("en")
    )
    respeller = EnglishRespeller() if use_english_respeller else None

    if phonetic_enabled and not use_english_respeller:
        logger.info(
            "Phonetic respelling is enabled but skipped: only English source language is supported."
        )

    srt_translator = None
    if model is not None:
        srt_translator = SRTTranslator(
            translator=model,
            respeller=respeller,
        )

    transcriber = WhisperTranscriber(language=whisper_language)
    speech_span_detector = None
    if speech_spans_enabled:
        speech_span_detector = SpeechSpanDetector(
            margin_sec=speech_spans_margin_sec,
            threshold=speech_vad_threshold,
            min_speech_duration_sec=speech_min_duration_sec,
            min_silence_duration_sec=speech_min_silence_sec,
            span_file_suffix=speech_span_file_suffix,
        )

    pipeline = VideoPipeline(
        video_folder=video_folder,
        translator=srt_translator,
        transcriber=transcriber,
        speech_span_detector=speech_span_detector,
        use_speech_spans_for_whisper=speech_spans_use_for_whisper,
        filter_sidecar_srt_by_speech_spans=speech_spans_filter_sidecar,
        detect_speech_spans_if_missing=speech_spans_detect_if_missing,
        overwrite_speech_spans=speech_spans_overwrite,
        single_video_name=single_video_name,
        speech_spans_only_mode=speech_spans_only_mode,
        speech_filter_rescue_dense_runs=speech_filter_rescue_dense_runs,
        speech_filter_rescue_min_cues=speech_filter_rescue_min_cues,
        speech_filter_rescue_min_duration_sec=speech_filter_rescue_min_duration_sec,
        speech_filter_rescue_max_duration_sec=speech_filter_rescue_max_duration_sec,
        speech_filter_rescue_min_density=speech_filter_rescue_min_density,
        speech_filter_rescue_tiny_runs=speech_filter_rescue_tiny_runs,
        speech_filter_rescue_tiny_max_cues=speech_filter_rescue_tiny_max_cues,
        speech_filter_rescue_tiny_max_duration_sec=speech_filter_rescue_tiny_max_duration_sec,
        speech_filter_rescue_tiny_neighbor_gap_sec=speech_filter_rescue_tiny_neighbor_gap_sec,
        sidecar_srt_encoding=sidecar_srt_encoding,
        output_folder=output_folder,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
