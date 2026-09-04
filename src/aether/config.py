"""Config dataclasses loadable from YAML files under configs/."""

from dataclasses import dataclass, field

import yaml


@dataclass
class AudioEncoderConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 800
    win_length: int = 800
    hop_length: int = 200

    conv_layers: list = field(default_factory=lambda: [[128, 5, 2], [256, 5, 2]])

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1

    qwen_hidden_size: int = 2048

    @classmethod
    def from_yaml(cls, path: str) -> "AudioEncoderConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
