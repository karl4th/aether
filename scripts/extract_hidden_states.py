"""Extract Qwen3-1.7B's last-layer hidden states for each Stage 2 sentence.

Reads data/dataset/sentences.jsonl (written by build_sentences.py), runs
each sentence through Qwen3-1.7B as normal text input, and saves the
full per-token last-layer hidden state sequence to
data/dataset/hidden_states/{id}.pt as a (seq_len, hidden_size) tensor.

Usage:
    python scripts/extract_hidden_states.py
    python scripts/extract_hidden_states.py --limit 10   # quick smoke test
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
SENTENCES_PATH = ROOT / "data" / "dataset" / "sentences.jsonl"
HIDDEN_STATES_DIR = ROOT / "data" / "dataset" / "hidden_states"
QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"


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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sentences = load_sentences(SENTENCES_PATH, args.limit)
    print(f"Loaded {len(sentences)} sentences from {SENTENCES_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading {QWEN_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, dtype=torch.float32)
    model.to(device)
    model.eval()

    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    skipped, extracted = 0, 0
    for record in tqdm(sentences, desc="Extracting hidden states"):
        out_path = HIDDEN_STATES_DIR / f"{record['id']}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        inputs = tokenizer(record["text"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0].cpu()  # (seq_len, hidden_size)

        torch.save(last_hidden, out_path)
        extracted += 1

    print(f"Extracted {extracted}, skipped {skipped} (already existed).")


if __name__ == "__main__":
    main()
