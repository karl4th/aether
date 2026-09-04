"""Run the trained Stage 3 decoder on a sentence and save the predicted
audio to a wav file so you can listen to it.

By default uses a sentence already in the Stage 2 dataset (fast, no Qwen
load needed). Pass --text to run arbitrary text through Qwen3-1.7B first.

Mel -> waveform uses Griffin-Lim (src/aether/audio/vocoder.py) since there's
no trained vocoder yet - expect robotic/buzzy quality, this is just to check
the decoder is producing something coherent.

Usage:
    python scripts/listen_to_decoder.py --id 00042
    python scripts/listen_to_decoder.py --text "The weather is nice today."
    python scripts/listen_to_decoder.py --id 00042 --checkpoint checkpoints/decoder/last.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aether.audio.vocoder import mel_to_waveform
from aether.config import DecoderConfig, TargetMelConfig
from aether.decoder.dataset import build_target_mel_fn
from aether.decoder.model import HiddenStateToMelDecoder

ROOT = Path(__file__).parent.parent
DECODER_CONFIG_PATH = ROOT / "configs" / "decoder.yaml"
TARGET_MEL_CONFIG_PATH = ROOT / "configs" / "target_mel.yaml"
DATASET_DIR = ROOT / "data" / "dataset"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "decoder" / "best.pt"
OUTPUT_DIR = ROOT / "data" / "samples"
QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"


def get_sentence_text(item_id: str) -> str:
    with open(DATASET_DIR / "sentences.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["id"] == item_id:
                return record["text"]
    raise ValueError(f"id {item_id} not found in sentences.jsonl")


def get_hidden_states_from_dataset(item_id: str, device: str) -> torch.Tensor:
    hidden_states = torch.load(DATASET_DIR / "hidden_states" / f"{item_id}.pt")
    return hidden_states.to(device)


def get_hidden_states_from_text(text: str, device: str) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {QWEN_MODEL_NAME} to compute hidden states for custom text...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, dtype=torch.float32)
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None, help="Sentence id from data/dataset/sentences.jsonl")
    parser.add_argument("--text", default=None, help="Arbitrary text (routes through Qwen live)")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--griffin-lim-iters", type=int, default=32)
    parser.add_argument("--out", default=None, help="Output wav path")
    parser.add_argument(
        "--vocoder-only",
        action="store_true",
        help=(
            "Skip the decoder entirely: round-trip the real target mel "
            "(from the actual Piper wav) through Griffin-Lim. Requires "
            "--id. Use this to check the vocoder's ceiling quality, "
            "independent of how well the decoder is trained."
        ),
    )
    args = parser.parse_args()

    if not args.id and not args.text:
        args.id = "00000"
    if args.id and args.text:
        raise ValueError("Pass either --id or --text, not both.")
    if args.vocoder_only and not args.id:
        raise ValueError("--vocoder-only requires --id (needs a real target wav).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mel_config = TargetMelConfig.from_yaml(str(TARGET_MEL_CONFIG_PATH))

    if args.vocoder_only:
        import soundfile as sf_read

        text = get_sentence_text(args.id)
        print(f"Text: {text!r}")
        audio_np, sr = sf_read.read(
            str(DATASET_DIR / "audio" / f"{args.id}.wav"), dtype="float32"
        )
        assert sr == mel_config.sample_rate
        waveform_in = torch.from_numpy(audio_np).unsqueeze(0)
        target_mel = build_target_mel_fn(mel_config)(waveform_in)[0].transpose(0, 1)
        print(f"Target mel shape: {tuple(target_mel.shape)}")

        waveform = mel_to_waveform(target_mel, mel_config, n_iter=args.griffin_lim_iters)
        out_path = Path(args.out) if args.out else OUTPUT_DIR / f"{args.id}_vocoder_ceiling.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform.numpy(), mel_config.sample_rate)
        print(f"Saved: {out_path}")
        print(f"Original audio for comparison: {DATASET_DIR / 'audio' / f'{args.id}.wav'}")
        return

    decoder_config = DecoderConfig.from_yaml(str(DECODER_CONFIG_PATH))
    decoder = HiddenStateToMelDecoder(decoder_config).to(device)
    decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    decoder.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.id:
        text = get_sentence_text(args.id)
        hidden_states = get_hidden_states_from_dataset(args.id, device)
        out_name = f"{args.id}.wav"
    else:
        text = args.text
        hidden_states = get_hidden_states_from_text(text, device)
        out_name = "custom_text.wav"

    print(f"Text: {text!r}")
    print(f"Hidden states shape: {tuple(hidden_states.shape)}")

    with torch.no_grad():
        pred_mel, log_length_pred = decoder(hidden_states, target_mel_len=None)
    predicted_len = round(torch.exp(log_length_pred).item())
    print(f"Predicted mel shape: {tuple(pred_mel.shape)} (predicted_len={predicted_len})")

    waveform = mel_to_waveform(pred_mel.cpu(), mel_config, n_iter=args.griffin_lim_iters)

    out_path = Path(args.out) if args.out else OUTPUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), waveform.numpy(), mel_config.sample_rate)
    print(f"Saved: {out_path}")

    if args.id:
        print(f"Ground-truth audio for comparison: {DATASET_DIR / 'audio' / f'{args.id}.wav'}")


if __name__ == "__main__":
    main()
