"""Griffin-Lim mel -> waveform inversion (placeholder vocoder, no learned params).

Used to listen to the Stage 3 decoder's predicted mel spectrograms before
any neural vocoder exists. Quality is mediocre (Griffin-Lim is a phase
estimation heuristic) but it's parameter-free and good enough to sanity
check whether the decoder is producing anything coherent.
"""

import torch
import torchaudio

from aether.config import TargetMelConfig


def mel_to_waveform(
    log_mel: torch.Tensor,
    mel_config: TargetMelConfig,
    n_iter: int = 32,
    log_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Args:
        log_mel: (mel_len, n_mels) log-compressed mel spectrogram, matching
            aether.audio.mel.LogMelSpectrogram's output convention
            (log(power_mel + log_eps)).
    Returns:
        waveform: (samples,) float32.
    """
    mel = (torch.exp(log_mel) - log_eps).clamp(min=0)
    mel = mel.transpose(0, 1)  # (n_mels, mel_len)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=mel_config.n_fft // 2 + 1,
        n_mels=mel_config.n_mels,
        sample_rate=mel_config.sample_rate,
    )
    linear_spec = inverse_mel(mel)  # (n_stft, mel_len), power spectrogram

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=mel_config.n_fft,
        win_length=mel_config.win_length,
        hop_length=mel_config.hop_length,
        n_iter=n_iter,
    )
    return griffin_lim(linear_spec)
