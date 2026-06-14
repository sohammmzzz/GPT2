# first_RNN — a character-level RNN language model in PyTorch

A small, fully-commented character-level recurrent language model. It reads text
one character at a time and learns to predict the next character; sampling from
it repeatedly produces new English-looking text.

Everything is split into small, single-responsibility modules so each stage of
the pipeline is easy to read and modify.

## Files

| File | What it does |
|------|--------------|
| `config.py`   | Every hyper-parameter and path in one `Config` dataclass. |
| `data.py`     | Download + clean public-domain text, char tokenizer, batching. |
| `model.py`    | The `CharRNN` model (LSTM / GRU / vanilla-RNN, configurable). |
| `train.py`    | Training loop: LR schedule, mixed precision, grad clipping, checkpoints. |
| `generate.py` | Sample text from a trained checkpoint (temperature + top-k). |
| `main.py`     | Run the whole thing end-to-end. |

## Quick start

```bash
# from the first_RNN directory, with the project venv active
python data.py          # (optional) just download + inspect the corpus
python main.py          # download -> train -> print a sample
```

Generate more text from the trained model:

```bash
python generate.py --prompt "Once upon a time" --length 1000 --temperature 0.8
```

## The data

By default the corpus is a handful of public-domain novels from Project
Gutenberg (Pride and Prejudice, Sherlock Holmes, A Tale of Two Cities,
Frankenstein, Alice in Wonderland), with the legal boilerplate stripped. This
gives fluent, modern-ish English prose. Switch to the classic tiny-shakespeare
corpus with `--dataset shakespeare`.

## The model

Three pieces: an **embedding** (char id → vector), a stacked **LSTM** (carries a
hidden "memory" state across the sequence), and a **linear head** (hidden state →
score for every possible next character). Defaults: 3 layers, hidden size 512,
~10–13M parameters — sized to train comfortably on a 6 GB GPU.

## Tuning for your GPU (RTX 4050, 6 GB)

The defaults in `config.py` already fit. If you hit out-of-memory, lower
`batch_size` or `seq_len`. To push quality higher with more VRAM headroom, raise
`hidden_size` or `num_layers`. Mixed precision (`use_amp`) is on by default for
speed and lower memory use.

## Reading the metrics

- **loss** — cross-entropy of next-char prediction (lower is better).
- **ppl (perplexity)** — `exp(loss)`; roughly how many characters the model is
  effectively choosing between. Good char-LMs on this kind of data reach single
  digits.
