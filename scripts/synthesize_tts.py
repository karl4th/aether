"""Synthesize target audio for each Stage 2 sentence using Piper TTS.

Reads data/dataset/sentences.jsonl (written by build_sentences.py) and
writes one wav file per sentence to data/dataset/audio/{id}.wav.

Usage:
    python scripts/synthesize_tts.py
    python scripts/synthesize_tts.py --limit 10   # quick smoke test
"""

import argparse
import json
import wave
from pathlib import Path

from piper.voice import PiperVoice
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
SENTENCES_PATH = ROOT / "data" / "dataset" / "sentences.jsonl"
AUDIO_DIR = ROOT / "data" / "dataset" / "audio"
DEFAULT_MODEL = ROOT / "data" / "piper_voices" / "en_US-lessac-medium.onnx"
DEFAULT_CONFIG = ROOT / "data" / "piper_voices" / "en_US-lessac-medium.onnx.json"


def load_sentences(path: Path, limit: int = None) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sentences = load_sentences(SENTENCES_PATH, args.limit)
    print(f"Loaded {len(sentences)} sentences from {SENTENCES_PATH}")

    voice = PiperVoice.load(args.model, args.config)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    skipped, synthesized = 0, 0
    for record in tqdm(sentences, desc="Synthesizing"):
        out_path = AUDIO_DIR / f"{record['id']}.wav"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize_wav(record["text"], wav_file)
        synthesized += 1

    print(f"Synthesized {synthesized}, skipped {skipped} (already existed).")


if __name__ == "__main__":
    main()
