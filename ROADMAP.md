# Roadmap

## Goal

Aether is a voice-to-voice pipeline built on **Qwen3-1.7B** with text
removed from the loop: audio goes in, audio comes out, no ASR/TTS text
stage in between. The open problem is decoding the model's internal
hidden state back into audio.

This file is the shared source of truth for scope and order. Edit it
directly — it's not just a Claude-side plan.

## Status

Stage 1 (audio encoder) and Stage 2 (dataset: 1500 sentence/audio/hidden-state
triples) are done. Stage 3 decoder architecture is done and smoke-tested;
training script is next, then the user runs training.

## Stage 1 — Audio

Get audio into Qwen. Turn a raw waveform into a sequence of embeddings in
Qwen's hidden size, fed to Qwen via `inputs_embeds` (no token/text path at
all).

Design: waveform (16kHz, mono) → log-mel spectrogram → Conv1D downsampling
stem → small Transformer encoder (~5-15M params) → linear projection to
Qwen's `hidden_size`. English-only for now.

- [x] `src/aether/audio/mel.py` — waveform → log-mel spectrogram
- [x] `src/aether/audio/encoder.py` — `AudioEncoder` module (conv stem +
      transformer + projection). Currently ~4.0M params (target 5-15M,
      can be scaled up later if needed).
- [x] `src/aether/config.py` + `configs/audio_encoder.yaml` — encoder
      hyperparameters
- [x] `scripts/smoke_test_audio_encoder.py` — dummy audio → encoder shape
      check passes: `(2, 61, 2048)`. Run with `--with-qwen` to also load
      Qwen3-1.7B for real (downloads weights, not run yet)

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

- [x] `scripts/build_sentences.py` — pull/filter/sample sentences from
      `wikitext-2-raw-v1`. 1500 sentences generated to
      `data/dataset/sentences.jsonl`.
- [x] `scripts/synthesize_tts.py` — Piper (`en_US-lessac-medium`, 22050Hz):
      sentence → target audio. 1500 wavs in `data/dataset/audio/{id}.wav`.
- [x] `scripts/extract_hidden_states.py` — Qwen3-1.7B (GPU): sentence text
      → last-layer hidden state sequence. 1500 tensors in
      `data/dataset/hidden_states/{id}.pt`, shape `(seq_len, 2048)`.
- [x] No separate manifest file needed — `sentences.jsonl`'s `id` field
      derives both paths by convention
      (`data/dataset/audio/{id}.wav`, `data/dataset/hidden_states/{id}.pt`).

## Stage 3 — Train and verify

Architecture: non-autoregressive, FastSpeech-style. `src/aether/decoder/model.py`
(`HiddenStateToMelDecoder`, `configs/decoder.yaml`, 10.6M params):
text-side transformer encodes Qwen's hidden states → length predictor
estimates mel frame count → length regulator stretches the sequence via
linear interpolation (no per-token alignment exists between Qwen's BPE
tokens and audio frames) → mel-side transformer decodes to a log-mel
spectrogram. Verified with `scripts/smoke_test_decoder.py` on real
dataset examples (teacher-forced and predicted-length forward passes
both run, shapes check out; length predictor is untrained so its output
is meaningless until trained).

- [x] Decoder architecture (`src/aether/decoder/model.py` +
      `configs/decoder.yaml`)
- [ ] Training script (mel L1 loss + length-predictor log-length loss,
      loops one example at a time over the Stage 2 dataset)
- [ ] Train (on GPU, run by the user)
- [ ] Verify: run held-out hidden states through the trained decoder,
      convert predicted mel to audio (e.g. Griffin-Lim), check it's
      correct/intelligible
