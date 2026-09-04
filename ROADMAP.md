# Roadmap

## Goal

Aether is a voice-to-voice pipeline built on **Qwen3-1.7B** with text
removed from the loop: audio goes in, audio comes out, no ASR/TTS text
stage in between. The open problem is decoding the model's internal
hidden state back into audio — a decoder analogous to a vocoder, but
taking Qwen's hidden representations as input instead of a
mel-spectrogram.

This file is the shared source of truth for scope and order. Edit it
directly — it's not just a Claude-side plan.

## Status

Stage 1 in progress.

## Stage 1 — Audio Input Encoder

Turn a raw waveform into a sequence of embeddings in Qwen's hidden size,
fed to Qwen via `inputs_embeds` (no token/text path at all).

Design: waveform (16kHz, mono) → log-mel spectrogram → Conv1D downsampling
stem → small Transformer encoder (~5-15M params) → linear projection to
Qwen's `hidden_size`. English-only for now.

- [ ] `src/aether/audio/mel.py` — waveform → log-mel spectrogram
- [ ] `src/aether/audio/encoder.py` — `AudioEncoder` module (conv stem +
      transformer + projection)
- [ ] `src/aether/config.py` + `configs/audio_encoder.yaml` — encoder
      hyperparameters
- [ ] `scripts/smoke_test_audio_encoder.py` — shape/param-count sanity
      check against Qwen3-1.7B's `hidden_size`

## Stage 2 — Feeding embeddings into Qwen3-1.7B

- [ ] Load Qwen3-1.7B, bypass its token embedding lookup
- [ ] Feed Stage 1's `inputs_embeds` directly into Qwen
- [ ] Confirm a forward pass runs and hidden states come out with the
      expected shape

## Stage 3 — Hidden state → audio decoder

This is where prior work stopped.

- [ ] Design/re-implement the module that turns Qwen's hidden states back
      into audio
- [ ] Reuse the idea of a small (~10M param), English-only transformer
      decoder for audio output

## Stage 4 — Training data & loss (TBD)

- [ ] Decide on English speech dataset(s) for training
- [ ] Define the end-to-end objective tying encoder, Qwen, and decoder
      together

## Stage 5 — End-to-end inference (TBD)

- [ ] Script that takes audio in (mic or file) and produces audio out
- [ ] Decide streaming vs. batch
