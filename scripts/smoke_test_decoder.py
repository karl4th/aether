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

import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aether.audio.mel import LogMelSpectrogram
from aether.config import DecoderConfig
from aether.decoder.model import HiddenStateToMelDecoder

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "decoder.yaml"
HIDDEN_STATES_DIR = ROOT / "data" / "dataset" / "hidden_states"
AUDIO_DIR = ROOT / "data" / "dataset" / "audio"

# Mel params matching Piper's 22050Hz output (see ROADMAP Stage 2).
TARGET_SAMPLE_RATE = 22050
TARGET_N_MELS = 80
TARGET_N_FFT = 1024
TARGET_WIN_LENGTH = 1024
TARGET_HOP_LENGTH = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="00000")
    args = parser.parse_args()

    config = DecoderConfig.from_yaml(str(CONFIG_PATH))
    decoder = HiddenStateToMelDecoder(config)
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"HiddenStateToMelDecoder parameter count: {num_params:,}")

    hidden_states = torch.load(HIDDEN_STATES_DIR / f"{args.id}.pt")
    print(f"Input hidden states shape: {tuple(hidden_states.shape)}")

    audio_np, sr = sf.read(str(AUDIO_DIR / f"{args.id}.wav"), dtype="float32")
    assert sr == TARGET_SAMPLE_RATE, f"expected {TARGET_SAMPLE_RATE}Hz, got {sr}Hz"
    waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)

    mel_fn = LogMelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mels=TARGET_N_MELS,
        n_fft=TARGET_N_FFT,
        win_length=TARGET_WIN_LENGTH,
        hop_length=TARGET_HOP_LENGTH,
    )
    target_mel = mel_fn(waveform)[0].transpose(0, 1)  # (mel_len, n_mels)
    target_mel_len = target_mel.shape[0]
    print(f"Target mel shape: {tuple(target_mel.shape)}")

    # Training mode: teacher-forced length.
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
