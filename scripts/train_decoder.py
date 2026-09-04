"""Train the Stage 3 decoder (Qwen hidden states -> mel spectrogram) on the
Stage 2 dataset.

One example at a time (the decoder isn't batched, see
src/aether/decoder/model.py), teacher-forced mel length during training.
Loss = mel L1 + weighted length-predictor L1 (in log-length space).

Usage:
    python scripts/train_decoder.py
    python scripts/train_decoder.py --epochs 50 --lr 3e-4
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aether.config import DecoderConfig, TargetMelConfig
from aether.decoder.dataset import Stage2Dataset
from aether.decoder.model import HiddenStateToMelDecoder

ROOT = Path(__file__).parent.parent
DECODER_CONFIG_PATH = ROOT / "configs" / "decoder.yaml"
TARGET_MEL_CONFIG_PATH = ROOT / "configs" / "target_mel.yaml"
DATASET_DIR = ROOT / "data" / "dataset"
CHECKPOINT_DIR = ROOT / "checkpoints" / "decoder"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def split_ids(all_ids: list, val_size: int, seed: int):
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    val_ids = sorted(ids[:val_size])
    train_ids = sorted(ids[val_size:])
    return train_ids, val_ids


def run_epoch(decoder, loader, device, length_loss_weight: float, optimizer=None):
    is_train = optimizer is not None
    decoder.train(is_train)

    total_mel_loss, total_length_loss, num_examples = 0.0, 0.0, 0
    for _item_id, hidden_states, target_mel in tqdm(loader, leave=False):
        hidden_states = hidden_states.to(device)
        target_mel = target_mel.to(device)

        with torch.set_grad_enabled(is_train):
            pred_mel, log_length_pred = decoder(
                hidden_states, target_mel_len=target_mel.shape[0]
            )
            mel_loss = F.l1_loss(pred_mel, target_mel)
            target_log_length = torch.log(
                torch.tensor(float(target_mel.shape[0]), device=device)
            )
            length_loss = F.l1_loss(log_length_pred, target_log_length)
            loss = mel_loss + length_loss_weight * length_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_mel_loss += mel_loss.item()
        total_length_loss += length_loss.item()
        num_examples += 1

    return total_mel_loss / num_examples, total_length_loss / num_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--length-loss-weight", type=float, default=0.1)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    decoder_config = DecoderConfig.from_yaml(str(DECODER_CONFIG_PATH))
    mel_config = TargetMelConfig.from_yaml(str(TARGET_MEL_CONFIG_PATH))

    full_dataset = Stage2Dataset(DATASET_DIR, mel_config)
    train_ids, val_ids = split_ids(full_dataset.ids, args.val_size, args.seed)
    print(f"Train: {len(train_ids)} sentences, Val: {len(val_ids)} sentences")

    train_dataset = Stage2Dataset(DATASET_DIR, mel_config, ids=train_ids)
    val_dataset = Stage2Dataset(DATASET_DIR, mel_config, ids=val_ids)

    collate_single = lambda batch: batch[0]  # noqa: E731 (batch_size=1, unwrap)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_single
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_single
    )

    decoder = HiddenStateToMelDecoder(decoder_config).to(device)
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"HiddenStateToMelDecoder parameter count: {num_params:,}")

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_mel_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mel_loss, train_length_loss = run_epoch(
            decoder, train_loader, device, args.length_loss_weight, optimizer
        )
        val_mel_loss, val_length_loss = run_epoch(
            decoder, val_loader, device, args.length_loss_weight, optimizer=None
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train mel {train_mel_loss:.4f} len {train_length_loss:.4f} | "
            f"val mel {val_mel_loss:.4f} len {val_length_loss:.4f}"
        )

        if val_mel_loss < best_val_mel_loss:
            best_val_mel_loss = val_mel_loss
            torch.save(decoder.state_dict(), CHECKPOINT_DIR / "best.pt")

        if epoch % args.checkpoint_every == 0:
            torch.save(decoder.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch:04d}.pt")

    torch.save(decoder.state_dict(), CHECKPOINT_DIR / "last.pt")
    print(f"Done. Best val mel loss: {best_val_mel_loss:.4f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
