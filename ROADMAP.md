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

Build a training dataset of ~1000-2000 paired examples: Qwen's real
internal hidden state for a sentence, and the audio that sentence should
produce.

Method:
1. Take ~1000-2000 English sentences from an existing text corpus
   (`wikitext-2-raw-v1` via HF `datasets`, sentence-split, filtered to
   natural spoken length, deduped, sampled).
2. Run each sentence through **Piper** (local TTS) to synthesize the
   target audio.
3. Run the same sentence text through Qwen3-1.7B as normal text input and
   capture the **full per-token hidden state sequence of the last layer**
   (`(seq_len, hidden_size)`) via `output_hidden_states=True`.
4. Store each pair (hidden states tensor + audio file) on disk with a
   manifest linking them by sentence id.

This does not depend on Stage 1's audio encoder — it uses Qwen in its
normal text mode to get hidden states, and Piper to get target audio.

- [ ] `scripts/build_sentences.py` — pull/filter/sample sentences from
      `wikitext-2-raw-v1`
- [ ] `scripts/synthesize_tts.py` — Piper: sentence → target audio (.wav)
- [ ] `scripts/extract_hidden_states.py` — Qwen3-1.7B: sentence text →
      last-layer hidden state sequence (.pt)
- [ ] `data/dataset/manifest.jsonl` (or similar) tying sentence id → audio
      path → hidden state path

## Stage 3 — Train and verify

- [ ] Train the hidden-state → audio decoder (small, ~10M param,
      English-only transformer, per the earlier module) on the Stage 2
      dataset
- [ ] Verify: run held-out hidden states through the trained decoder and
      check the output audio is correct/intelligible
