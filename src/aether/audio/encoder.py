"""Audio input encoder: waveform -> sequence of embeddings in Qwen's hidden size."""

import torch
from torch import nn

from aether.audio.mel import LogMelSpectrogram
from aether.config import AudioEncoderConfig
from aether.layers import SinusoidalPositionalEncoding


class ConvStem(nn.Module):
    """Strided Conv1D blocks that downsample mel frames and lift channels to d_model."""

    def __init__(self, n_mels: int, conv_layers: list, d_model: int):
        super().__init__()
        blocks = []
        in_channels = n_mels
        for out_channels, kernel_size, stride in conv_layers:
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            blocks.append(nn.GELU())
            in_channels = out_channels
        blocks.append(nn.Conv1d(in_channels, d_model, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (batch, n_mels, frames) -> (batch, d_model, frames_down)"""
        return self.net(mel)


class AudioEncoder(nn.Module):
    """Waveform -> (batch, seq_len, qwen_hidden_size) embeddings for `inputs_embeds`."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.mel = LogMelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
        )
        self.conv_stem = ConvStem(config.n_mels, config.conv_layers, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_proj = nn.Linear(config.d_model, config.qwen_hidden_size)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples), float32.
        Returns:
            embeddings: (batch, seq_len, qwen_hidden_size)
        """
        mel = self.mel(waveform)  # (batch, n_mels, frames)
        x = self.conv_stem(mel)  # (batch, d_model, frames_down)
        x = x.transpose(1, 2)  # (batch, frames_down, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.output_proj(x)
