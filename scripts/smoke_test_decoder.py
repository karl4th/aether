"""Sanity-check the Stage 3 decoder: shapes and param count on one real
hidden-state/audio pair from the Stage 2 dataset.

Usage:
    python scripts/smoke_test_decoder.py
    python scripts/smoke_test_decoder.py --id 00007
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aether.config import DecoderConfig, TargetMelConfig
from aether.decoder.dataset import Stage2Dataset
from aether.decoder.model import HiddenStateToMelDecoder

ROOT = Path(__file__).parent.parent
DECODER_CONFIG_PATH = ROOT / "configs" / "decoder.yaml"
TARGET_MEL_CONFIG_PATH = ROOT / "configs" / "target_mel.yaml"
DATASET_DIR = ROOT / "data" / "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="00000")
    args = parser.parse_args()

    config = DecoderConfig.from_yaml(str(DECODER_CONFIG_PATH))
    decoder = HiddenStateToMelDecoder(config)
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"HiddenStateToMelDecoder parameter count: {num_params:,}")

    mel_config = TargetMelConfig.from_yaml(str(TARGET_MEL_CONFIG_PATH))
    dataset = Stage2Dataset(DATASET_DIR, mel_config, ids=[args.id])
    item_id, hidden_states, target_mel = dataset[0]

    print(f"Input hidden states shape: {tuple(hidden_states.shape)}")
    print(f"Target mel shape: {tuple(target_mel.shape)}")

    # Training mode: teacher-forced length.
    target_mel_len = target_mel.shape[0]
    pred_mel, log_length_pred = decoder(hidden_states, target_mel_len=target_mel_len)
    print(f"Predicted mel shape (teacher-forced length): {tuple(pred_mel.shape)}")
    assert pred_mel.shape == target_mel.shape

    predicted_len = round(torch.exp(log_length_pred).item())
    print(
        f"Length predictor: log_length_pred={log_length_pred.item():.3f} "
        f"-> predicted_len={predicted_len} (untrained, target was {target_mel_len})"
    )

    # Inference mode: predicted length (untrained, so just checking it runs).
    pred_mel_infer, _ = decoder(hidden_states, target_mel_len=None)
    print(f"Predicted mel shape (predicted length): {tuple(pred_mel_infer.shape)}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
