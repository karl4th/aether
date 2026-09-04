"""Build the Stage 2 sentence list from wikitext-2-raw-v1.

Pulls plain English text, splits into sentences, filters to natural
spoken length, dedupes, and samples a fixed-size set with a fixed seed
so the output is reproducible.

Usage:
    python scripts/build_sentences.py --num-sentences 1500
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "dataset" / "sentences.jsonl"

HEADER_RE = re.compile(r"^\s*=+.*=+\s*$")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
WORD_RE = re.compile(r"\S+")


SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:%)\]])")
SPACE_AROUND_APOSTROPHE_RE = re.compile(r"\s+([’'])\s*s\b")


def clean_wikitext_line(line: str) -> str:
    """Undo wikitext-2-raw-v1's tokenization artifacts around punctuation."""
    line = line.replace(" @-@ ", "-")
    line = line.replace(" @.@ ", ".")
    line = line.replace(" @,@ ", ",")
    line = SPACE_AROUND_APOSTROPHE_RE.sub(r"\1s", line)
    line = SPACE_BEFORE_PUNCT_RE.sub(r"\1", line)
    return line.strip()


def is_clean_sentence(sentence: str) -> bool:
    if not sentence or HEADER_RE.match(sentence):
        return False
    if sentence[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ\"'":
        return False
    if sentence[-1] not in ".!?":
        return False
    if "=" in sentence or "@" in sentence:
        return False
    num_words = len(WORD_RE.findall(sentence))
    return 4 <= num_words <= 20


def extract_sentences(raw_lines) -> list:
    sentences = []
    for raw_line in raw_lines:
        line = clean_wikitext_line(raw_line)
        if not line or HEADER_RE.match(line):
            continue
        for candidate in SENTENCE_END_RE.split(line):
            candidate = candidate.strip()
            if is_clean_sentence(candidate):
                sentences.append(candidate)
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sentences", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading wikitext-2-raw-v1...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    print("Extracting and filtering sentences...")
    sentences = extract_sentences(ds["text"])
    sentences = sorted(set(sentences))  # dedupe, deterministic order before sampling
    print(f"Found {len(sentences)} candidate sentences after filtering/dedup.")

    if len(sentences) < args.num_sentences:
        print(
            f"WARNING: only {len(sentences)} sentences available, "
            f"fewer than requested {args.num_sentences}."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(sentences, min(args.num_sentences, len(sentences)))
    sampled.sort()  # stable, readable output order

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sampled):
            record = {"id": f"{i:05d}", "text": sentence}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} sentences to {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
