"""
Transcribe video files using Whisper and process SRT files with translation.

Directory structure for each video:
video_folder/
├── video_file.mp4
├── video_file.srt (duplicate in DUPLICATE_SRT_ENCODING)
└── video_file_name/
    ├── video_file.srt (original, utf-8)
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

import nltk
import torch
from dotenv import load_dotenv
from g2p_en import G2p
from transformers import MarianMTModel, MarianTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

for pkg in [
    "averaged_perceptron_tagger_eng",
    "cmudict",
    "punkt",
]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

load_dotenv()

VIDEO_FOLDER = Path(os.getenv("VIDEO_FOLDER", ""))
DUPLICATE_SRT_ENCODING = os.getenv("DUPLICATE_SRT_ENCODING", "utf-8")
LANGUAGE = os.getenv("LANGUAGE", "en")
HF_TOKEN = os.getenv("HF_TOKEN", None)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", None)
if OUTPUT_FOLDER:
    OUTPUT_FOLDER = Path(OUTPUT_FOLDER)

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

PHONEME_MAP_EN = {
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

g2p = G2p()


def respell_english(text: str) -> str:
    phonemes = g2p(text)

    words = []
    current = []

    for p in phonemes:
        if p == " ":
            if current:
                words.append(_respell_word(current))
                current = []
            continue

        current.append(p)

    if current:
        words.append(_respell_word(current))

    return " ".join(words)


def _respell_word(phonemes: list[str]) -> str:
    syllables = []
    current = []

    for p in phonemes:
        base = re.sub(r"\d", "", p)

        if base in VOWELS and current:
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
            letters = PHONEME_MAP_EN.get(base, base)

            if stress:
                letters = letters.upper()
            else:
                letters = letters.lower()

            parts.append(letters)

        rendered.append("".join(parts))

    return "-".join(rendered)


def load_model(model_name: str, hf_token: str | None = None):
    tokenizer = MarianTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model = MarianMTModel.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer, model) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Ensure pad_token_id is set correctly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_srt(
    input_path: Path,
    output_path: Path,
    tokenizer,
    model,
    mode: str = "append",
    batch_size: int = 32,
    include_pronunciation: bool = False,
    use_english_respeller: bool = True,
) -> None:
    if mode not in {"replace", "append"}:
        raise ValueError("Неизвестный режим перевода")

    lines = input_path.read_text(encoding="utf-8").splitlines()

    timestamp_pattern = re.compile(r"\d\d:\d\d:\d\d,\d\d\d")

    text_lines = []
    text_indices = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if (
            stripped.isdigit()
            or "-->" in line
            or not stripped
            or timestamp_pattern.search(line)
        ):
            continue

        text_lines.append(stripped)
        text_indices.append(i)

    translations = []

    for i in range(0, len(text_lines), batch_size):
        batch = text_lines[i : i + batch_size]
        translations.extend(translate_batch(batch, tokenizer, model))

    translation_map = dict(zip(text_indices, translations))

    out_lines = []

    for i, line in enumerate(lines):
        if i not in translation_map:
            out_lines.append(line)
            continue

        translated = translation_map[i]

        if mode == "replace":
            out_lines.append(translated)

        elif mode == "append":
            out_lines.append(line)

            if include_pronunciation:
                if use_english_respeller:
                    out_lines.append(respell_english(line))
                else:
                    out_lines.append(respell_sentence(line))

            out_lines.append(translated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines), encoding="utf-8")


def respell_sentence(text: str) -> str:
    phonemes = g2p(text)

    words = []
    current = []

    for p in phonemes:
        if p == " ":
            if current:
                words.append(phonemes_to_respelling(current))
                current = []
            continue

        current.append(p)

    if current:
        words.append(phonemes_to_respelling(current))

    return " ".join(words)


def phonemes_to_respelling(phonemes: list[str]) -> str:
    out = []

    for p in phonemes:
        p = re.sub(r"\d", "", p)

        if p in PHONEME_MAP_EN:
            out.append(PHONEME_MAP_EN[p])
        else:
            out.append(p.lower())

    return "".join(out)


def transcribe_video(video_path: Path) -> tuple[Path, str] | None:
    """
    Transcribe a video file using Whisper CLI.

    Returns:
        Tuple of (srt_file_path, stdout_output) or None if already exists
    """
    video_folder = video_path.parent
    video_name = video_path.stem

    # Create output directory for this video
    output_dir = video_folder / video_name
    output_dir.mkdir(exist_ok=True)

    # Output paths - use video name for SRT file
    srt_path = output_dir / f"{video_name}.srt"
    stdout_path = output_dir / "stdout.txt"

    # Skip if already transcribed
    if srt_path.exists() and stdout_path.exists():
        logger.info(f"  Transcription already exists, skipping Whisper")
        return srt_path, stdout_path.read_text(encoding="utf-8")

    # Run whisper command
    cmd = [
        "whisper",
        str(video_path),
        "--language",
        LANGUAGE,
        "--output_format",
        "srt",
        "--fp16",
        "False",
        "--verbose",
        "True",
    ]

    logger.info("Starting Whisper transcription...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Execute with streaming stdout for real-time progress monitoring
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=video_folder,
        bufsize=1,
        env=env,
    )

    stdout_lines = []

    if process.stdout:
        while True:
            line = process.stdout.readline()
            if not line:
                break

            line = line.rstrip()
            stdout_lines.append(line)
            logger.info("Whisper: %s", line)

    process.wait()

    # Combine stdout for saving to file
    stdout_content = "\n".join(stdout_lines)

    # Whisper outputs SRT file with same name as video, move it to our output dir
    whisper_srt_path = video_path.with_suffix(".srt")
    if whisper_srt_path.exists():
        whisper_srt_path.rename(srt_path)

    # Write stdout to file
    stdout_path.write_text(stdout_content, encoding="utf-8")

    if process.returncode != 0:
        logger.error(f"Whisper failed with return code {process.returncode}")

    return srt_path, stdout_content


def process_srt_file(
    video_path: Path,
    srt_path: Path,
    output_dir: Path,
    tokenizer,
    model,
    use_english_respeller: bool,
) -> tuple[Path, Path] | None:
    """
    Process SRT file: translate and save in different encodings.

    Returns:
        Tuple of (translated_windows1251_path, translated_utf8_path) or None if already exists
    """
    video_name = video_path.stem

    # Create translated directories
    translated_dir_windows1251 = output_dir / "translated_windows1251"
    translated_dir_utf8 = output_dir / "translated_utf8"

    translated_dir_windows1251.mkdir(exist_ok=True)
    translated_dir_utf8.mkdir(exist_ok=True)

    # Output paths for translated files (use video name)
    translated_windows1251_path = translated_dir_windows1251 / f"{video_name}.srt"
    translated_utf8_path = translated_dir_utf8 / f"{video_name}.srt"

    # Skip if already translated
    if translated_windows1251_path.exists() and translated_utf8_path.exists():
        logger.info(f"  Translation already exists, skipping")
        return translated_windows1251_path, translated_utf8_path

    # Translate and save in UTF-8 first
    translate_srt(
        input_path=srt_path,
        output_path=translated_utf8_path,
        tokenizer=tokenizer,
        model=model,
        mode="append",
        batch_size=32,
        include_pronunciation=True,
        use_english_respeller=use_english_respeller,
    )

    # Read translated UTF-8 content and save as Windows-1251
    translated_content = translated_utf8_path.read_text(encoding="utf-8")
    translated_windows1251_path.write_text(translated_content, encoding="windows-1251")

    return translated_windows1251_path, translated_utf8_path


def create_duplicate_in_video_folder(
    video_path: Path, translated_utf8_path: Path, encoding: str
) -> Path:
    """
    Create a duplicate of the translated SRT file in the video folder.

    Returns:
        Path to the duplicate file
    """
    video_name = video_path.stem
    content = translated_utf8_path.read_text(encoding="utf-8")
    duplicate_path = video_path.parent / f"{video_name}.srt"
    duplicate_path.write_text(content, encoding=encoding)
    return duplicate_path


def main():
    if not VIDEO_FOLDER.exists():
        logger.error(f"Video folder does not exist: {VIDEO_FOLDER}")
        return

    # Load translation model
    model_name = "Helsinki-NLP/opus-mt-en-ru"
    logger.info(f"Loading translation model: {model_name}")
    tokenizer, model = load_model(model_name, hf_token=HF_TOKEN)

    # Determine respeller mode based on LANGUAGE env var
    use_english_respeller = LANGUAGE.lower() == "en"

    # Get list of videos
    video_files = list(VIDEO_FOLDER.glob("*.mp4"))
    total_videos = len(video_files)

    if not video_files:
        logger.warning(f"No .mp4 files found in {VIDEO_FOLDER}")
        return

    logger.info(f"Found {total_videos} video(s) to process")

    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Video {idx}/{total_videos}: {video_path.name}")
        logger.info(f"{'='*60}")

        video_name = video_path.stem
        output_dir = video_path.parent / video_name

        # Check what needs to be done
        srt_path = output_dir / f"{video_name}.srt"
        stdout_path = output_dir / "stdout.txt"
        translated_windows1251_path = output_dir / "translated_windows1251" / f"{video_name}.srt"
        translated_utf8_path = output_dir / "translated_utf8" / f"{video_name}.srt"
        duplicate_path_in_video = video_path.parent / f"{video_name}.srt"

        transcription_exists = srt_path.exists() and stdout_path.exists()
        translation_exists = translated_windows1251_path.exists() and translated_utf8_path.exists()
        duplicate_exists = duplicate_path_in_video.exists()

        # Skip if everything is already done
        if transcription_exists and translation_exists and duplicate_exists:
            logger.info(f"Skipping {video_path.name}: all outputs already exist")

            # Move to output folder if specified
            if OUTPUT_FOLDER:
                _move_to_output_folder(video_path, output_dir, OUTPUT_FOLDER)

            continue

        logger.info(f"Processing: {video_path.name}")

        # Step 1: Transcribe (if needed)
        if transcription_exists:
            logger.info(f"  Transcription already exists, skipping Whisper")
        else:
            srt_path, stdout = transcribe_video(video_path)
            logger.info(f"  Created SRT: {srt_path}")

        # Step 2: Translate (if needed)
        if translation_exists:
            logger.info(f"  Translation already exists, skipping")
        else:
            translated_windows1251_path, translated_utf8_path = process_srt_file(
                video_path,
                srt_path,
                output_dir,
                tokenizer,
                model,
                use_english_respeller,
            )
            logger.info(f"  Created translated files:")
            logger.info(f"    {translated_windows1251_path}")
            logger.info(f"    {translated_utf8_path}")

        # Step 3: Create duplicate in video folder (if needed)
        if duplicate_exists:
            logger.info(f"  Duplicate already exists, skipping")
        else:
            duplicate_path = create_duplicate_in_video_folder(
                video_path, translated_utf8_path, DUPLICATE_SRT_ENCODING
            )
            logger.info(
                f"  Created duplicate: {duplicate_path} (encoding: {DUPLICATE_SRT_ENCODING})"
            )

        logger.info(f"  Done!\n")

        # Move to output folder if specified
        if OUTPUT_FOLDER:
            _move_to_output_folder(video_path, output_dir, OUTPUT_FOLDER)


def _move_to_output_folder(video_path: Path, output_dir: Path, output_folder: Path) -> None:
    """Move video and its output directory to the output folder."""
    video_name = video_path.stem

    # Create destination directory
    dest_dir = output_folder / video_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Move video file
    dest_video = dest_dir / f"{video_name}.mp4"
    if video_path.exists() and not dest_video.exists():
        shutil.move(str(video_path), str(dest_video))
        logger.info(f"  Moved video to: {dest_video}")

    # Move output directory (subtitles, translations, etc.)
    if output_dir.exists():
        for item in output_dir.iterdir():
            dest_item = dest_dir / item.name
            if not dest_item.exists():
                shutil.move(str(item), str(dest_item))
        # Remove empty output directory
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
        logger.info(f"  Moved output directory to: {dest_dir}")


if __name__ == "__main__":
    main()
