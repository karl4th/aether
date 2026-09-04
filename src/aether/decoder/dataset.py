"""Loads the Stage 2 dataset (sentences.jsonl + audio/{id}.wav +
hidden_states/{id}.pt) as (hidden_states, target_mel) pairs."""

import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from aether.audio.mel import LogMelSpectrogram
from aether.config import TargetMelConfig


def build_target_mel_fn(config: TargetMelConfig) -> LogMelSpectrogram:
    return LogMelSpectrogram(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
    )


class Stage2Dataset(Dataset):
    """One item = one sentence: (id, hidden_states (seq_len_text, hidden_size),
    target_mel (mel_len, n_mels))."""

    def __init__(self, dataset_dir: Path, mel_config: TargetMelConfig, ids: list = None):
        self.dataset_dir = Path(dataset_dir)
        self.mel_fn = build_target_mel_fn(mel_config)
        self.sample_rate = mel_config.sample_rate

        if ids is not None:
            self.ids = ids
        else:
            self.ids = []
            with open(self.dataset_dir / "sentences.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    self.ids.append(json.loads(line)["id"])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        item_id = self.ids[index]
        hidden_states = torch.load(self.dataset_dir / "hidden_states" / f"{item_id}.pt")

        audio_np, sr = sf.read(
            str(self.dataset_dir / "audio" / f"{item_id}.wav"), dtype="float32"
        )
        assert sr == self.sample_rate, f"expected {self.sample_rate}Hz, got {sr}Hz"
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
        target_mel = self.mel_fn(waveform)[0].transpose(0, 1)  # (mel_len, n_mels)

        return item_id, hidden_states, target_mel
