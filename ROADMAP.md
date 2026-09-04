# Roadmap

## Goal

Aether is a voice-to-voice pipeline built on **Qwen3-1.7B** with text
removed from the loop: audio goes in, audio comes out, no ASR/TTS text
stage in between. The open problem is decoding the model's internal
hidden state back into audio.

This file is the shared source of truth for scope and order. Edit it
directly — it's not just a Claude-side plan.

## Status

Stage 1 in progress.

## Stage 1 — Audio

Get audio into Qwen. Turn a raw waveform into a sequence of embeddings in
Qwen's hidden size, fed to Qwen via `inputs_embeds` (no token/text path at
all).

Design: waveform (16kHz, mono) → log-mel spectrogram → Conv1D downsampling
stem → small Transformer encoder (~5-15M params) → linear projection to
Qwen's `hidden_size`. English-only for now.

- [ ] `src/aether/audio/mel.py` — waveform → log-mel spectrogram
- [ ] `src/aether/audio/encoder.py` — `AudioEncoder` module (conv stem +
      transformer + projection)
- [ ] `src/aether/config.py` + `configs/audio_encoder.yaml` — encoder
      hyperparameters
- [ ] `scripts/smoke_test_audio_encoder.py` — feed real/dummy audio through
      Qwen3-1.7B (via `inputs_embeds`) and confirm hidden states come out
      with the expected shape

## Stage 2 — Dataset: hidden states → target audio

Build a training dataset of paired examples: Qwen's real internal hidden
state for a given input, and the audio that should be produced from it.

- [ ] Run real audio through Stage 1's encoder + Qwen3-1.7B to collect
      actual hidden states (not synthetic/dummy ones)
- [ ] Pair each hidden state with its target audio
- [ ] **Open question to settle before building this:** what counts as
      "target audio" for a given hidden state — the same audio reconstructed
      (autoencoder-style), or a separate response audio (real voice-to-voice
      pairs)? This decides what raw data we need to collect/use.
- [ ] Store pairs in a dataset format usable for training (e.g. on-disk
      tensors/shards of hidden states + matching audio)

## Stage 3 — Train and verify

- [ ] Train the hidden-state → audio decoder (small, ~10M param,
      English-only transformer, per the earlier module) on the Stage 2
      dataset
- [ ] Verify: run held-out hidden states through the trained decoder and
      check the output audio is correct/intelligible
