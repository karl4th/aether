"""Waveform to log-mel spectrogram front-end."""

import torch
import torchaudio
from torch import nn


class LogMelSpectrogram(nn.Module):
    """Converts a raw waveform into a log-compressed mel spectrogram."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 800,
        win_length: int = 800,
        hop_length: int = 200,
        log_eps: float = 1e-5,
    ):
        super().__init__()
        self.log_eps = log_eps
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples), float32, range [-1, 1].
        Returns:
            log_mel: (batch, n_mels, frames)
        """
        mel = self.mel_spec(waveform)
        return torch.log(mel + self.log_eps)
