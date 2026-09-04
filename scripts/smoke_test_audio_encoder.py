"""Sanity-check the Stage 1 audio encoder: shapes, param count, and (optionally)
a real forward pass through Qwen3-1.7B via inputs_embeds.

Usage:
    python scripts/smoke_test_audio_encoder.py
    python scripts/smoke_test_audio_encoder.py --with-qwen   # also loads Qwen3-1.7B (downloads weights)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aether.audio.encoder import AudioEncoder
from aether.config import AudioEncoderConfig

QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "audio_encoder.yaml")


def run_encoder_only(config: AudioEncoderConfig) -> torch.Tensor:
    encoder = AudioEncoder(config)
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"AudioEncoder parameter count: {num_params:,}")

    batch_size, seconds = 2, 3
    dummy_waveform = torch.randn(batch_size, seconds * config.sample_rate)
    embeddings = encoder(dummy_waveform)
    print(f"Input waveform shape:  {tuple(dummy_waveform.shape)}")
    print(f"Output embedding shape: {tuple(embeddings.shape)}")

    assert embeddings.shape[0] == batch_size
    assert embeddings.shape[2] == config.qwen_hidden_size
    return embeddings


def check_qwen_hidden_size(config: AudioEncoderConfig) -> None:
    from transformers import AutoConfig

    qwen_config = AutoConfig.from_pretrained(QWEN_MODEL_NAME)
    print(f"{QWEN_MODEL_NAME} hidden_size: {qwen_config.hidden_size}")
    if qwen_config.hidden_size != config.qwen_hidden_size:
        print(
            f"WARNING: configs/audio_encoder.yaml qwen_hidden_size="
            f"{config.qwen_hidden_size} does not match the real model "
            f"({qwen_config.hidden_size}). Update the yaml file."
        )
    else:
        print("qwen_hidden_size in configs/audio_encoder.yaml matches the real model.")


def run_with_qwen(config: AudioEncoderConfig, embeddings: torch.Tensor) -> None:
    from transformers import AutoModelForCausalLM

    print(f"Loading {QWEN_MODEL_NAME} (this downloads weights on first run)...")
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME)
    model.eval()

    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]
    print(f"Qwen last hidden state shape: {tuple(last_hidden.shape)}")
    assert last_hidden.shape[:2] == embeddings.shape[:2]
    assert last_hidden.shape[2] == model.config.hidden_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-qwen",
        action="store_true",
        help="Also load Qwen3-1.7B and run a real forward pass (downloads weights).",
    )
    args = parser.parse_args()

    config = AudioEncoderConfig.from_yaml(CONFIG_PATH)
    embeddings = run_encoder_only(config)

    if args.with_qwen:
        check_qwen_hidden_size(config)
        run_with_qwen(config, embeddings)
    else:
        print("Skipping Qwen forward pass (pass --with-qwen to run it).")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
