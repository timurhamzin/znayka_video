"""
Transcribe video files using Whisper and process SRT files with translation.

Directory structure for each video:
video_folder/
├── video_file.mp4
├── video_file.srt (translated, in DUPLICATE_SRT_ENCODING)
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
from pathlib import Path

# Disable HuggingFace and Transformers network access
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

    def __init__(self, model_name: str, hf_token: str | None = None):
        self.tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.model = MarianMTModel.from_pretrained(
            model_name,
            local_files_only=True,
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

    def transcribe(self, video_path: Path, output_dir: Path) -> tuple[Path, str]:
        video_name = video_path.stem

        # Create original subfolder
        original_dir = output_dir / "original"
        original_dir.mkdir(exist_ok=True)

        srt_path = original_dir / f"{video_name}.srt"
        stdout_path = output_dir / "stdout.txt"

        if srt_path.exists() and stdout_path.exists():
            logger.info("Transcription already exists")
            return srt_path, stdout_path.read_text(encoding="utf-8")

        cmd = [
            "whisper",
            str(video_path),
            "--language",
            self.language,
            "--output_format",
            "srt",
        ]

        logger.info("Starting Whisper transcription...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=video_path.parent,
            bufsize=1,
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
            logger.error("Whisper failed with return code %s", process.returncode)

        return srt_path, stdout_content


class VideoPipeline:
    """Responsible for orchestrating the video processing workflow."""

    def __init__(
            self,
            video_folder: Path,
            translator: SRTTranslator,
            transcriber: WhisperTranscriber,
            duplicate_encoding: str = "utf-8",
            output_folder: Path | None = None,
    ):
        self.video_folder = video_folder
        self.translator = translator
        self.transcriber = transcriber
        self.duplicate_encoding = duplicate_encoding
        self.output_folder = output_folder

    def run(self) -> None:
        if not self.video_folder.exists():
            logger.error("Video folder does not exist: %s", self.video_folder)
            return

        if self.output_folder:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info("Output folder: %s", self.output_folder)

        videos = list(self.video_folder.glob("*.mp4"))

        if not videos:
            logger.warning("No .mp4 files found in %s", self.video_folder)
            return

        logger.info("Found %d video(s) to process", len(videos))

        for idx, video in enumerate(videos, 1):
            logger.info("")
            logger.info("=" * 60)
            logger.info("Video %d/%d: %s", idx, len(videos), video.name)
            logger.info("=" * 60)

            self._process_video(video)

    def _process_video(self, video: Path) -> None:
        name = video.stem
        output_dir = video.parent / name

        output_dir.mkdir(exist_ok=True)

        original_srt = output_dir / "original" / f"{name}.srt"
        stdout_path = output_dir / "stdout.txt"
        translated_windows1251 = output_dir / "translated_windows1251" / f"{name}.srt"
        translated_utf8 = output_dir / "translated_utf8" / f"{name}.srt"
        duplicate_path = video.parent / f"{name}.srt"

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
            srt_path, _ = self.transcriber.transcribe(video, output_dir)
            logger.info("  Created SRT: %s", srt_path)

        if translation_exists:
            logger.info("  Translation already exists, skipping")
        else:
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
                self.duplicate_encoding,
            )

        logger.info("  Done!")

        if self.output_folder:
            self._move_to_output_folder(video, output_dir)

    def _duplicate(self, video: Path, srt: Path) -> None:
        content = srt.read_text(encoding="utf-8")
        duplicate = video.parent / f"{video.stem}.srt"
        duplicate.write_text(content, encoding=self.duplicate_encoding, errors='replace')

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

    duplicate_encoding = _first_env(
        "TRANSCRIBE_DUPLICATE_SRT_ENCODING",
        "DUPLICATE_SRT_ENCODING",
        default="utf-8",
    )

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

    if translation_model_name is None:
        translation_model_name = resolve_translation_model(
            translation_source_language,
            translation_target_language,
        )

    if output_folder:
        output_folder = Path(output_folder)

    model = TranslationModel(translation_model_name, hf_token=hf_token)

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

    srt_translator = SRTTranslator(
        translator=model,
        respeller=respeller,
    )

    transcriber = WhisperTranscriber(language=whisper_language)

    pipeline = VideoPipeline(
        video_folder=video_folder,
        translator=srt_translator,
        transcriber=transcriber,
        duplicate_encoding=duplicate_encoding,
        output_folder=output_folder,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
