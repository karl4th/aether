# Aether

A voice-to-voice project built on PyTorch and the **Qwen3-1.7B** model.

## Idea

Text is removed from the pipeline: instead of the classic
`audio → text (ASR) → LLM → text → audio (TTS)` scheme, the model works
directly with speech, using text only for training/intermediate
supervision (if at all).

The core challenge is decoding the model's **internal hidden state**
back into audio — i.e. a decoder module that turns Qwen's hidden
representations into an audio signal (similar to a vocoder, but the
input is the LLM's internal state rather than a mel-spectrogram).

## Structure (draft)

```
aether/
├── src/            # pipeline source code
├── configs/        # experiment configs
├── scripts/        # helper scripts (training, inference)
├── notebooks/       # experiment notebooks
├── docs/           # architecture notes and documentation
└── data/           # data (gitignored, not stored in git)
```

## Model

- Base LLM: `Qwen/Qwen3-1.7B`
- Framework: PyTorch
