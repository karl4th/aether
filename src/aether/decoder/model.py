"""Stage 3 decoder: Qwen hidden states -> mel spectrogram.

Non-autoregressive, FastSpeech-style: a text-side transformer encodes
Qwen's per-token hidden states, a length predictor estimates how many mel
frames the utterance should take, a length regulator stretches the
encoded sequence to that many frames (linear interpolation over time —
Qwen's BPE tokens have no per-token alignment to audio frames, unlike
phoneme-level TTS durations), and a mel-side transformer decodes the
stretched sequence into mel frames.

Operates on one example at a time (no batching): hidden-state and mel
lengths vary a lot across the dataset's ~1500 sentences, and padding/
masking for batched variable-length interpolation isn't worth the
complexity yet.
"""

import torch
import torch.nn.functional as F
from torch import nn

from aether.config import DecoderConfig
from aether.layers import SinusoidalPositionalEncoding


class LengthRegulator(nn.Module):
    """Stretches a (seq_len, d_model) sequence to (target_len, d_model) by
    linear interpolation over the time axis."""

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = x.transpose(0, 1).unsqueeze(0)  # (1, d_model, seq_len)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1)  # (target_len, d_model)


class HiddenStateToMelDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.qwen_hidden_size, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model)

        text_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            text_layer, num_layers=config.num_text_encoder_layers
        )

        self.length_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.length_regulator = LengthRegulator()

        mel_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.mel_decoder = nn.TransformerEncoder(
            mel_layer, num_layers=config.num_mel_decoder_layers
        )
        self.output_proj = nn.Linear(config.d_model, config.n_mels)

    def forward(self, hidden_states: torch.Tensor, target_mel_len: int = None):
        """
        Args:
            hidden_states: (seq_len_text, qwen_hidden_size) — Qwen's
                last-layer hidden states for one sentence.
            target_mel_len: ground-truth mel frame count. When given, the
                length regulator stretches to this exact length (teacher
                forcing, used during training). When None, the predicted
                length is used instead (inference).
        Returns:
            mel: (mel_len, n_mels) predicted log-mel spectrogram.
            log_length_pred: scalar predicted log(mel_len), for the
                length-predictor loss during training.
        """
        x = self.input_proj(hidden_states).unsqueeze(0)  # (1, seq_len_text, d_model)
        x = self.pos_encoding(x)
        x = self.text_encoder(x).squeeze(0)  # (seq_len_text, d_model)

        pooled = x.mean(dim=0)  # (d_model,)
        log_length_pred = self.length_predictor(pooled).squeeze(-1)  # scalar

        if target_mel_len is not None:
            mel_len = target_mel_len
        else:
            # Floor at 8 frames: below this, downstream mel->waveform
            # inversion (Griffin-Lim needs mel_len * hop_length >= n_fft)
            # breaks. Only matters for a degenerate/undertrained predictor;
            # real utterances are always far longer.
            mel_len = max(8, round(torch.exp(log_length_pred).item()))

        x = self.length_regulator(x, mel_len)  # (mel_len, d_model)
        x = self.mel_decoder(x.unsqueeze(0)).squeeze(0)  # (mel_len, d_model)
        mel = self.output_proj(x)  # (mel_len, n_mels)

        return mel, log_length_pred
