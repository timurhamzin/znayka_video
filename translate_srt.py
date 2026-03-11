from pathlib import Path
import os
import re

import torch
from transformers import MarianMTModel, MarianTokenizer
from g2p_en import G2p

import nltk

for pkg in [
    'averaged_perceptron_tagger_eng',
    'cmudict',
    'punkt',
]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

g2p = G2p()

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW"
}



PHONEME_MAP = {
    "AA": "A",
    "AE": "A",
    "AH": "U",
    "AO": "AW",
    "AW": "OW",
    "AY": "AY",
    "B": "B",
    "CH": "CH",
    "D": "D",
    "DH": "TH",
    "EH": "E",
    "ER": "ER",
    "EY": "AY",
    "F": "F",
    "G": "G",
    "HH": "H",
    "IH": "I",
    "IY": "EE",
    "JH": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "NG": "NG",
    "OW": "OH",
    "OY": "OY",
    "P": "P",
    "R": "R",
    "S": "S",
    "SH": "SH",
    "T": "T",
    "TH": "TH",
    "UH": "U",
    "UW": "OO",
    "V": "V",
    "W": "W",
    "Y": "Y",
    "Z": "Z",
    "ZH": "ZH",
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


def phonemes_to_respelling(phonemes: list[str]) -> str:
    """Преобразует фонемы в упрощённую запись."""
    out = []

    for p in phonemes:
        p = re.sub(r'\d', '', p)

        if p in PHONEME_MAP:
            out.append(PHONEME_MAP[p])
        else:
            out.append(p.lower())

    return ''.join(out)


def respell_sentence(text: str) -> str:
    """Создаёт упрощённую фонетическую запись."""
    phonemes = g2p(text)

    words = []
    current = []

    for p in phonemes:

        if p == ' ':
            if current:
                words.append(phonemes_to_respelling(current))
                current = []
            continue

        current.append(p)

    if current:
        words.append(phonemes_to_respelling(current))

    return ' '.join(words)

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


def load_model(model_name: str):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


def translate_batch(texts: list[str], tokenizer, model) -> list[str]:

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_srt(
    input_path: Path,
    output_path: Path,
    tokenizer,
    model,
    mode: str = 'append',
    batch_size: int = 32,
    include_pronunciation: bool = False,
    use_english_respeller: bool = True,
) -> None:

    if mode not in {'replace', 'append'}:
        raise ValueError('Неизвестный режим перевода')

    lines = input_path.read_text(encoding='utf-8').splitlines()

    timestamp_pattern = re.compile(r'\d\d:\d\d:\d\d,\d\d\d')

    text_lines = []
    text_indices = []

    for i, line in enumerate(lines):

        stripped = line.strip()

        if (
            stripped.isdigit()
            or '-->' in line
            or not stripped
            or timestamp_pattern.search(line)
        ):
            continue

        text_lines.append(stripped)
        text_indices.append(i)

    translations = []

    for i in range(0, len(text_lines), batch_size):
        batch = text_lines[i:i + batch_size]
        translations.extend(translate_batch(batch, tokenizer, model))

    translation_map = dict(zip(text_indices, translations))

    out_lines = []

    for i, line in enumerate(lines):

        if i not in translation_map:
            out_lines.append(line)
            continue

        translated = translation_map[i]

        if mode == 'replace':
            out_lines.append(translated)

        elif mode == 'append':

            out_lines.append(line)

            if include_pronunciation:
                if use_english_respeller:
                    out_lines.append(respell_english(line))
                else:
                    out_lines.append(respell_sentence(line))

            out_lines.append(translated)

    output_path.write_text('\n'.join(out_lines), encoding='utf-8')


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    model_name = 'Helsinki-NLP/opus-mt-en-ru'

    tokenizer, model = load_model(model_name)

    # Выбор функции транслитерации на основе переменной окружения LANGUAGE
    # 'en' - использует respell_english (с разбиением на слоги и ударениями)
    # другое значение - использует respell_sentence (простая транслитерация)
    language = os.environ.get('LANGUAGE', 'en')
    use_english_respeller = language.lower() == 'en'

    input_folder = Path(os.environ.get('INPUT_SRT_FOLDER', ''))
    output_folder = Path(os.environ.get('OUTPUT_SRT_FOLDER', ''))

    if not input_folder.exists():
        print(f'Input folder does not exist: {input_folder}')
        exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    srt_files = list(input_folder.glob('*.srt'))

    if not srt_files:
        print(f'No .srt files found in {input_folder}')
        exit(1)

    print(f'Found {len(srt_files)} .srt file(s) to process')

    for srt_file in srt_files:
        output_path = output_folder / srt_file.name
        print(f'Processing: {srt_file.name} -> {output_path}')

        translate_srt(
            input_path=srt_file,
            output_path=output_path,
            tokenizer=tokenizer,
            model=model,
            mode='append',
            batch_size=32,
            include_pronunciation=True,
            use_english_respeller=use_english_respeller,
        )

    print('Done!')