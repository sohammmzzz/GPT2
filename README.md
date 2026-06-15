# First_RNN_1B — a conversational RNN with Memory Caching

A from-scratch **recurrent** language model that aims for transformer-level
quality by implementing **Memory Caching** (Behrouz et al., *"Memory Caching:
RNNs with Growing Memory"*, 2026). You **train on Colab** (free 16 GB GPU) and
**chat locally** on your 6 GB GPU through a Streamlit UI.

> Folder name note: Windows is case-insensitive, so a literal `First_RNN` would
> collide with the existing `first_RNN` char-RNN project. This project lives in
> `First_RNN_1B` to keep them separate.

---

## What's in here

| File | Purpose |
|------|---------|
| `build_corpus.py` | Data pipeline. Downloads **real** public text (Project Gutenberg) + **real** conversations (Dolly-15k, OpenAssistant) into `data/corpus.txt` and `data/chat.jsonl`. No fabricated data. |
| `train.py` | **Everything**: the model architecture, both training stages (pre-train → conversational fine-tune), checkpoint/resume, and weight export. Run this on Colab. |
| `inference.py` | Streamlit chat UI. Loads `weights/model_final.pt` and runs on your local GPU. |
| `data/` | `corpus.txt` (12.8 MB) + `chat.jsonl` (22,348 conversations) — already built. |
| `weights/` | **You** drop the trained `model_final.pt` here after Colab. |
| `requirements-colab.txt` | pip deps for Colab. |

## The model

A recurrent mixer (constant-size state, `O(L)` like a true RNN) made
competitive with attention via three ideas from the 2025–26 literature:

1. **Gated Linear Attention core** — a linear-attention RNN with a
   data-dependent forget gate on its state (the Mamba/GLA idea).
2. **Memory Caching (GRM variant)** — the sequence is split into segments;
   each segment is compressed into its own small state; every token reads from
   **all past segment states** through a **context-aware gated** sum. This is
   what gives the recurrent model a memory that *grows with the sequence* —
   previously attention's exclusive trick — and closes the recall gap.
3. **Hybrid layers** — every 6th layer is full global attention (with RoPE).
   The paper proves attention is itself Memory Caching at segment-size-1.

Plus standard modern scaffolding: RMSNorm, SwiGLU MLPs, weight-tied embeddings,
tiktoken GPT-2 BPE (50,257 vocab).

| Preset (`--size`) | Total params | d_model × layers |
|-------------------|--------------|------------------|
| `125m` | 129.5 M | 768 × 12 |
| `350m` (default) | **354.3 M** | 1024 × 22 |
| `1b` | 0.984 B | 2048 × 16 |

---

## How to run it (the full loop)

### 1. (Optional) rebuild/grow the local data
Already built for you. To make a bigger pre-training corpus:
```bash
python build_corpus.py --max_books 15 --max_oasst 8000
```

### 2. Train on Colab (free T4, 16 GB)
Upload `train.py` (and, if you want the bigger streaming corpus, nothing else —
it downloads data itself). In a Colab cell:

```python
!pip install torch tiktoken datasets requests
# 350M model, streaming FineWeb-Edu for pretraining, then chat fine-tune:
!python train.py --size 350m --source hf
```

- `--source hf` streams large web datasets (FineWeb-Edu → OpenWebText → WikiText
  fallback) so you are **not** limited by upload size.
- `--source local` instead uses the `data/` files (upload the `data/` folder).
- **Colab disconnects every ~12 h** — just rerun the same command; it
  **auto-resumes** from `checkpoints/` (optimizer state included).
- For the `1b` preset add `--grad_checkpoint` and lower `--batch_size`.

When it finishes you get `weights/model_final.pt`.

### 3. Chat locally
Download `model_final.pt` into `First_RNN_1B/weights/`, then:
```bash
pip install streamlit torch tiktoken
streamlit run inference.py
```
A 350 M model in fp16 is ~0.7 GB — it runs comfortably on your 6 GB RTX 4050.

---

## Key training flags

```
--size {125m,350m,1b}     model size (default 350m)
--source {local,hf}       data source (default local; use hf on Colab)
--pretrain_steps N        default 20000
--finetune_steps N        default 3000
--batch_size / --grad_accum   effective batch = batch_size * grad_accum
--grad_checkpoint         trade compute for VRAM (use for 1b)
--skip_pretrain           jump straight to fine-tuning (resumes pretrained ckpt)
```

---

## Honest expectations

This is a genuine, modern architecture trained on **free** hardware — calibrate
accordingly:

- A from-scratch model trained for a few Colab sessions will produce **fluent,
  on-topic conversational English**, but it is **not** GPT-4. It will hallucinate
  facts and lose hard reasoning. That's expected at this scale/compute, not a bug.
- **More pre-training tokens = better.** The single biggest quality lever is
  letting Stage 1 run longer (raise `--pretrain_steps`).
- The Memory-Caching mechanism most improves **long-context recall**, which is
  where plain RNNs fail worst — that's the whole point of the paper.
- Generation in `inference.py` recomputes the context each token (so Memory
  Caching is fully active and identical to training). Fine for chat lengths; a
  streaming recurrent cache is a future optimization.
