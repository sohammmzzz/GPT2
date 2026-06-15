"""
train.py — A conversational RNN with Memory Caching (Behrouz et al., 2026)
==========================================================================

ONE file containing:
  * the full model architecture (Gated Linear Attention recurrent core +
    GRM Memory Caching + a few interleaved global-attention layers),
  * a self-contained data pipeline (local files OR direct web download),
  * STAGE 1: language-model PRE-TRAINING,
  * STAGE 2: conversational SUPERVISED FINE-TUNING (chat),
  * checkpoint/resume (essential for free Colab's ~12h sessions),
  * final weight export to ./weights/ for the Streamlit `inference.py`.

Designed to be copied to a Google Colab notebook (free T4, 16 GB) and run:

    !pip install torch tiktoken datasets requests
    !python train.py --size 350m --source hf

It will pre-train, then fine-tune, then write ./weights/model_final.pt .

----------------------------------------------------------------------------
WHY THIS IS A RECURRENT MODEL THAT RIVALS TRANSFORMERS
----------------------------------------------------------------------------
The mixer is a *linear-attention RNN* (constant-size state, O(L) like an RNN),
upgraded with two ideas from the literature:
  1. a data-dependent forget GATE on the state (Gated Linear Attention / Mamba),
  2. MEMORY CACHING (this paper): the sequence is split into segments, each
     segment is compressed into its own small state, and every token reads from
     ALL past segment-states through a context-aware GATED aggregation (GRM).
     This gives the recurrent model a memory that GROWS with the sequence —
     the property that previously only attention had — closing the recall gap.
A few full-attention layers are interleaved (hybrid), which the paper shows is
itself just Memory Caching at segment-size-1.
"""

from __future__ import annotations

import os
# Reduce CUDA fragmentation OOMs on memory-tight GPUs (e.g. Colab T4).
# Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import time
import urllib.request
import gzip
import io
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
CKPT_DIR = ROOT / "checkpoints"


os.environ["TRITON_PTXAS_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\ptxas.exe"

# Fix 2: Disable CUDA graphs — Windows C long is 32-bit,
# CUDA stream handles overflow it (PyTorch issue #162430)
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"

# Fix 3: Updated allocator config (old env var is deprecated)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------- #
#  Hugging Face access token (for streaming datasets with --source hf)
# --------------------------------------------------------------------------- #
# Preferred: set it in your shell so it NEVER lands in source control:
#     Windows PowerShell:  $env:HF_TOKEN="hf_xxx"
#     Linux/macOS:         export HF_TOKEN=hf_xxx
# Fallback (plug-and-play): drop your token into a file named `hf_token.txt`
# next to this script — it's git-ignored, so it never lands in source control.
HF_TOKEN = os.environ.get("HF_TOKEN", "") or ""  # <-- or paste here: "hf_xxxxxxxx"
if not HF_TOKEN:
    _tok_file = Path(__file__).resolve().parent / "hf_token.txt"
    if _tok_file.exists():
        HF_TOKEN = _tok_file.read_text(encoding="utf-8").strip()
if HF_TOKEN:
    # Make the token visible to every HF library (datasets, hub) automatically.
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN



# =========================================================================== #
#  TOKENIZER  (tiktoken GPT-2 BPE — 50257 vocab, fast & well-tested)
# =========================================================================== #
# Chat is encoded as PLAIN TEXT with role markers. They tokenize into ordinary
# BPE tokens (no custom vocab needed) and the real <|endoftext|> id ends a turn.

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
DEFAULT_SYSTEM = "You are a helpful, friendly assistant."


class ChatTokenizer:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token  # 50256
        self.vocab_size = self.enc.n_vocab  # 50257

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids if 0 <= int(i) < self.vocab_size]
        return self.enc.decode(ids)

    # -- chat formatting --------------------------------------------------- #
    def build_prompt(self, messages: list[dict], system: str = DEFAULT_SYSTEM) -> str:
        """Render messages -> the text the model is trained to continue."""
        s = f"{SYSTEM_TAG}\n{system}\n"
        for m in messages:
            if m["role"] == "user":
                s += f"{USER_TAG}\n{m['content']}\n"
            else:
                s += f"{ASSISTANT_TAG}\n{m['content']}{self.enc.decode([self.eot])}"
        return s

    def build_inference_prompt(self, messages: list[dict], system: str = DEFAULT_SYSTEM) -> list[int]:
        """Prompt ending right after the assistant tag, ready for generation."""
        s = f"{SYSTEM_TAG}\n{system}\n"
        for m in messages:
            if m["role"] == "user":
                s += f"{USER_TAG}\n{m['content']}\n"
            else:
                s += f"{ASSISTANT_TAG}\n{m['content']}{self.enc.decode([self.eot])}"
        s += f"{ASSISTANT_TAG}\n"
        return self.encode(s)

    def build_sft_example(self, messages: list[dict], system: str = DEFAULT_SYSTEM,
                          max_len: int = 1024) -> tuple[list[int], list[int]]:
        """
        Return (input_ids, labels). Labels mask everything except the assistant
        responses (and their terminating <|endoftext|>) with -100, so the loss is
        only computed on what the assistant should say.
        """
        ids: list[int] = []
        labels: list[int] = []

        def add(text: str, supervise: bool):
            toks = self.encode(text)
            ids.extend(toks)
            labels.extend(toks if supervise else [-100] * len(toks))

        add(f"{SYSTEM_TAG}\n{system}\n", False)
        for m in messages:
            if m["role"] == "user":
                add(f"{USER_TAG}\n{m['content']}\n", False)
            else:
                add(f"{ASSISTANT_TAG}\n", False)
                add(m["content"], True)
                ids.append(self.eot)
                labels.append(self.eot)  # learn to stop
        return ids[:max_len], labels[:max_len]


# =========================================================================== #
#  MODEL CONFIG
# =========================================================================== #

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 22
    n_heads: int = 16
    mlp_hidden: int = 2816
    segment_size: int = 64          # Memory-Caching segment length C
    attn_every: int = 6             # every Nth layer is full global attention (hybrid)
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


SIZE_PRESETS = {
    # name : (d_model, n_layers, n_heads, mlp_hidden)
    "125m": (768, 12, 12, 2048),
    "350m": (1024, 22, 16, 2816),
    "1b":   (2048, 16, 16, 5632),
}


def make_config(size: str, vocab_size: int, max_seq_len: int, segment_size: int) -> ModelConfig:
    d, l, h, m = SIZE_PRESETS[size]
    return ModelConfig(vocab_size=vocab_size, d_model=d, n_layers=l, n_heads=h,
                       mlp_hidden=m, max_seq_len=max_seq_len, segment_size=segment_size)


# =========================================================================== #
#  BUILDING BLOCKS
# =========================================================================== #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward, the standard modern transformer MLP."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Precompute rotary position embedding cos/sin tables."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)         # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None] + rot * sin[None, None]


class GatedLinearAttentionMC(nn.Module):
    """
    The recurrent mixer with Memory Caching (GRM variant).

    Pipeline per head:
      1. Split the T tokens into N segments of length C.
      2. Within each segment: causal gated linear attention (intra-segment).
      3. Compress each segment into its own state matrix S_m  (d x d).
      4. Cross-segment read: every query segment reads from ALL past segment
         states through a CONTEXT-AWARE GATED sum (GRM) — this is the growing
         memory. Gate weight between query-segment s and key-segment m is
         softmax_m < meanpool_u(S_s), meanpool_k(S_m) >, with u a SEPARATE
         projection from q (the paper's ablation shows sharing them collapses).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.n_heads
        self.Dh = cfg.head_dim
        self.C = cfg.segment_size
        D = cfg.d_model
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.u_proj = nn.Linear(D, D, bias=False)      # MC gating projection (≠ q)
        self.g_proj = nn.Linear(D, self.H, bias=True)  # per-head scalar forget gate
        self.o_proj = nn.Linear(D, D, bias=False)
        self.head_norm = RMSNorm(self.Dh)
        # init forget-gate bias high -> decay≈0.98 at start (slow forgetting)
        nn.init.constant_(self.g_proj.bias, 4.0)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh, C = self.H, self.Dh, self.C
        # pad time to a multiple of C
        pad = (C - T % C) % C
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        N = Tp // C

        q = self.q_proj(x).view(B, Tp, H, Dh).transpose(1, 2)   # (B,H,Tp,Dh)
        k = self.k_proj(x).view(B, Tp, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, Tp, H, Dh).transpose(1, 2)
        u = self.u_proj(x).view(B, Tp, H, Dh).transpose(1, 2)
        # feature map keeps linear-attention scores non-negative & stable
        q = F.silu(q)
        k = F.silu(k)
        # per-head log-decay in fp32 for numerical stability
        g = torch.sigmoid(self.g_proj(x).float()).transpose(1, 2)  # (B,H,Tp) in (0,1)
        la = torch.log(g.clamp(min=1e-6))                          # <=0

        # reshape into segments: (B,H,N,C,Dh)
        def seg(t):
            return t.reshape(B, H, N, C, t.shape[-1])
        qs, ks, vs, us = seg(q), seg(k), seg(v), seg(u)
        las = la.reshape(B, H, N, C)                  # (B,H,N,C)
        A = torch.cumsum(las, dim=-1)                 # inclusive cumulative log-decay
        A_end = A[..., -1:]                           # (B,H,N,1) total per segment

        # ---- 1. intra-segment causal gated linear attention --------------- #
        # weight[i,j] = (q_i·k_j) * exp(A_i - A_j) for j<=i
        qk = torch.einsum("bhncd,bhned->bhnce", qs.float(), ks.float())  # (B,H,N,C,C) [i,e=j]
        decay = (A.unsqueeze(-1) - A.unsqueeze(-2))                      # (B,H,N,C,C) [i,j]=A_i-A_j
        causal = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
        # IMPORTANT: mask BEFORE exp. In the upper triangle A_i - A_j > 0, so
        # exp() can overflow to +inf as the forget gate sharpens during training.
        # masked_fill AFTER exp hides the inf in the forward pass but the backward
        # pass through exp(inf) yields NaN gradients -> weights become NaN.
        # Setting masked positions to -inf first makes exp -> 0 with zero gradient.
        decay = decay.masked_fill(~causal, float("-inf"))
        scores = qk * torch.exp(decay)
        o_intra = torch.einsum("bhnce,bhned->bhncd", scores, vs.float())  # (B,H,N,C,Dh)

        # ---- 2. per-segment compressed state S_m -------------------------- #
        # S_m = sum_j exp(A_end - A_j) k_j ⊗ v_j      (independent compressor)
        kd = ks.float() * torch.exp(A_end - A).unsqueeze(-1)            # (B,H,N,C,Dh)
        S = torch.einsum("bhncd,bhnce->bhnde", kd, vs.float())          # (B,H,N,Dh,Dh)

        # ---- 3. context-aware gates between segments (GRM) ---------------- #
        # meanpool of keys per segment (the segment's content summary)
        mp = ks.float().mean(dim=3)                    # (B,H,N,Dh)
        uq = us.float().mean(dim=3)                    # query-segment summary (≠ q)
        rel = torch.einsum("bhsd,bhmd->bhsm", uq, mp) / math.sqrt(Dh)   # (B,H,N,N)
        seg_idx = torch.arange(N, device=x.device)
        cross_mask = seg_idx[None, :] < seg_idx[:, None]                # m < s
        # Use a large FINITE negative (not -inf) so the first segment's fully
        # masked row doesn't produce an all-(-inf) softmax -> NaN (forward AND
        # backward). After softmax we zero the masked columns, so segment 0 ends
        # up with WS=0 cleanly, with no NaN anywhere in the graph.
        neg = torch.finfo(rel.dtype).min
        rel = rel.masked_fill(~cross_mask[None, None], neg)
        gamma = torch.softmax(rel, dim=-1)                             # (B,H,N,N)
        gamma = gamma.masked_fill(~cross_mask[None, None], 0.0)        # drop masked contributions

        # weighted sum of past segment states, then read with the query
        WS = torch.einsum("bhsm,bhmde->bhsde", gamma, S)              # (B,H,N,Dh,Dh)
        o_cross = torch.einsum("bhscd,bhsde->bhsce", qs.float(), WS)   # (B,H,N,C,Dh)

        # ---- combine, normalise, project --------------------------------- #
        o = (o_intra + o_cross).reshape(B, H, Tp, Dh)
        o = self.head_norm(o.to(x.dtype))
        o = o.transpose(1, 2).reshape(B, Tp, D)
        o = self.o_proj(o)
        return o[:, :T]


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (the 'hybrid' layers) with RoPE."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.n_heads
        self.Dh = cfg.head_dim
        D = cfg.d_model
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.H, self.Dh
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        cos, sin = rope_cache(T, Dh, self.cfg.rope_base, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(o)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_attention: bool):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = CausalSelfAttention(cfg) if use_attention else GatedLinearAttentionMC(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RNNLM(nn.Module):
    """The full language model."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg, use_attention=((i + 1) % cfg.attn_every == 0))
            for i in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.grad_checkpoint = False
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None and not torch.allclose(m.bias, torch.zeros_like(m.bias)):
                pass
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel()  # don't double-count tied head

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eot_id, temperature=0.8, top_k=50,
                 top_p=0.95, repetition_penalty=1.1, context_window=1024):
        """Recompute-based autoregressive generation (reuses the training forward,
        so Memory Caching is fully active). O(L) per token — fine for chat."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -context_window:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                for t in set(idx[0].tolist()):
                    logits[0, t] /= repetition_penalty
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                sp[cum - sp > top_p] = 0.0
                sp /= sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            else:
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if int(nxt) == eot_id:
                break
        return idx


# =========================================================================== #
#  DATA PIPELINE  (local files if present, else download)
# =========================================================================== #

def _http_text(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


class PretrainData:
    """Yields fixed-length token blocks for next-token pre-training.

    `--source local` : tokenize ./data/corpus.txt (built by build_corpus.py).
    `--source hf`    : stream HuggingFaceFW/fineweb-edu on the fly (Colab).
    """
    def __init__(self, tok: ChatTokenizer, seq_len: int, source: str, device):
        self.tok = tok
        self.seq_len = seq_len
        self.source = source
        self.device = device
        self._buf: list[int] = []
        if source == "local":
            path = DATA_DIR / "corpus.txt"
            assert path.exists(), "data/corpus.txt missing — run build_corpus.py"
            print("[data] tokenizing local corpus.txt ...")
            text = path.read_text(encoding="utf-8")
            self._tokens = torch.tensor(tok.encode(text), dtype=torch.long)
            print(f"[data] pretrain tokens: {len(self._tokens):,}")
        else:
            self._tokens = None
            self._stream = self._hf_stream()

    def _hf_stream(self) -> Iterator[list[int]]:
        """Yield already-tokenized documents (each a list of ids ending in EOT).
        Yielding whole docs and bulk-extending the buffer is far faster than the
        per-token append it replaces — token-at-a-time Python is the bottleneck."""
        from datasets import load_dataset
        for name in [("HuggingFaceFW/fineweb-edu", "sample-10BT"),
                     ("Skylion007/openwebtext", None),
                     ("wikitext", "wikitext-103-raw-v1")]:
            try:
                ds = load_dataset(name[0], name[1], split="train", streaming=True,
                                  token=HF_TOKEN or None)
                print(f"[data] streaming {name[0]} ...")
                for row in ds:
                    txt = row.get("text") or ""
                    if txt.strip():
                        toks = self.tok.encode(txt)
                        toks.append(self.tok.eot)
                        yield toks
            except Exception as e:  # noqa: BLE001
                print(f"[data] {name[0]} unavailable ({e}); trying next")
        raise RuntimeError("No HF pretraining dataset available")

    def get_batch(self, batch_size: int):
        need = batch_size * self.seq_len + 1
        if self.source == "local":
            ix = torch.randint(0, len(self._tokens) - self.seq_len - 1, (batch_size,))
            x = torch.stack([self._tokens[i:i + self.seq_len] for i in ix])
            y = torch.stack([self._tokens[i + 1:i + 1 + self.seq_len] for i in ix])
        else:
            while len(self._buf) < need:
                self._buf.extend(next(self._stream))
            flat = torch.tensor(self._buf[:need], dtype=torch.long)
            self._buf = self._buf[batch_size * self.seq_len:]
            x = flat[:-1].view(batch_size, self.seq_len)
            y = flat[1:].view(batch_size, self.seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
             "resolve/main/databricks-dolly-15k.jsonl")
OASST_TREES_URL = ("https://huggingface.co/datasets/OpenAssistant/oasst1/"
                   "resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz")


def load_chat_records(source: str) -> list[dict]:
    """Return a list of {"messages":[...]} conversations from local or web."""
    local = DATA_DIR / "chat.jsonl"
    if source == "local" or local.exists():
        if local.exists():
            print("[data] loading local chat.jsonl ...")
            recs = [json.loads(l) for l in local.read_text(encoding="utf-8").splitlines() if l.strip()]
            return recs
    # download dolly + oasst directly
    recs: list[dict] = []
    try:
        print("[data] downloading dolly-15k ...")
        for line in _http_text(DOLLY_URL).splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ins, ctx, resp = (r.get("instruction") or "").strip(), (r.get("context") or "").strip(), (r.get("response") or "").strip()
            if ins and resp:
                user = ins if not ctx else f"{ins}\n\n{ctx}"
                recs.append({"messages": [{"role": "user", "content": user},
                                          {"role": "assistant", "content": resp}]})
    except Exception as e:  # noqa: BLE001
        print(f"[data] dolly failed: {e}")
    try:
        print("[data] downloading oasst1 trees ...")
        req = urllib.request.Request(OASST_TREES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
        text = gzip.GzipFile(fileobj=io.BytesIO(blob)).read().decode("utf-8", "ignore")

        def walk(node, prefix, out):
            role = "user" if node.get("role") == "prompter" else "assistant"
            t = (node.get("text") or "").strip()
            chain = prefix + [{"role": role, "content": t}] if t else prefix
            if role == "assistant" and len(chain) >= 2:
                out.append(chain)
            for c in (node.get("replies") or []):
                walk(c, chain, out)

        for line in text.splitlines():
            if not line.strip():
                continue
            tree = json.loads(line)
            prompt = tree.get("prompt")
            if not prompt or prompt.get("lang") not in (None, "en"):
                continue
            chains = []
            walk(prompt, [], chains)
            for ch in chains[:2]:
                recs.append({"messages": ch})
    except Exception as e:  # noqa: BLE001
        print(f"[data] oasst failed: {e}")
    if not recs:
        raise RuntimeError("No chat data available")
    print(f"[data] loaded {len(recs):,} conversations")
    return recs


class ChatData:
    """Tokenizes chat records into (input, label) pairs with prompt-masking and
    packs them into fixed-length batches."""
    def __init__(self, tok: ChatTokenizer, records: list[dict], seq_len: int, device):
        self.tok, self.seq_len, self.device = tok, seq_len, device
        self.examples = []
        for r in records:
            ids, labels = tok.build_sft_example(r["messages"], max_len=seq_len)
            if len(ids) >= 8:
                self.examples.append((ids, labels))
        print(f"[data] SFT examples: {len(self.examples):,}")

    def get_batch(self, batch_size: int):
        import random
        xs, ys = [], []
        for _ in range(batch_size):
            ids, labels = random.choice(self.examples)
            pad = self.seq_len - len(ids)
            x = ids + [self.tok.eot] * pad
            y = labels + [-100] * pad
            xs.append(x[:self.seq_len])
            ys.append(y[:self.seq_len])
        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y


# =========================================================================== #
#  TRAINING LOOP (shared by both stages)
# =========================================================================== #

def get_lr(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def run_stage(model, data, *, stage, steps, batch_size, grad_accum, base_lr,
              warmup, device, amp_dtype, grad_clip, log_every, save_every,
              ckpt_path, start_step=0):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    # restore optimizer state if resuming this stage
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        if ck.get("stage") == stage and "opt" in ck:
            opt.load_state_dict(ck["opt"])
            start_step = ck["step"]
            print(f"[{stage}] resumed optimizer at step {start_step}")

    t0 = time.time()
    running = 0.0
    for step in range(start_step, steps):
        lr = get_lr(step, steps, base_lr, warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(grad_accum):
            x, y = data.get_batch(batch_size)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_val += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        running += loss_val

        if (step + 1) % log_every == 0:
            avg = running / log_every
            running = 0.0
            tok_s = log_every * batch_size * grad_accum * data.seq_len / (time.time() - t0)
            t0 = time.time()
            print(f"[{stage}] step {step+1}/{steps} | lr {lr:.2e} | loss {avg:.4f} "
                  f"| ppl {math.exp(min(avg,20)):.2f} | {tok_s/1e3:.1f}k tok/s")

        if (step + 1) % save_every == 0:
            save_ckpt(ckpt_path, model, opt, stage, step + 1)
            print(f"[{stage}] checkpoint saved at step {step+1}")

    save_ckpt(ckpt_path, model, opt, stage, steps)


def _raw_model(model):
    """Unwrap a torch.compile()'d model so state_dict keys have no `_orig_mod.`
    prefix. Saving the compiled wrapper's state_dict embeds that prefix, which
    then fails to load into a plain RNNLM (e.g. in inference.py)."""
    return getattr(model, "_orig_mod", model)


def _atomic_save(obj, path: Path):
    """Write to a temp file then atomically replace — avoids half-written/
    truncated checkpoints if the process or disk hiccups mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_ckpt(path, model, opt, stage, step):
    """Full resumable checkpoint (model + optimizer, fp32) — stays on Colab."""
    raw = _raw_model(model)
    _atomic_save({"model": raw.state_dict(), "opt": opt.state_dict() if opt else None,
                  "stage": stage, "step": step, "cfg": asdict(raw.cfg)}, path)


def export_weights(path, model):
    """Lightweight, downloadable weights: model only, fp16 (half the size).
    `inference.py` rebuilds the model from `cfg` and loads these."""
    raw = _raw_model(model)
    sd = {k: v.detach().half().cpu() for k, v in raw.state_dict().items()}
    _atomic_save({"model": sd, "cfg": asdict(raw.cfg), "tokenizer": "gpt2"}, path)


# =========================================================================== #
#  MAIN — pretrain, then finetune, then export weights
# =========================================================================== #

def main():
    p = argparse.ArgumentParser(description="Train a Memory-Caching conversational RNN")
    p.add_argument("--size", default="350m", choices=list(SIZE_PRESETS))
    p.add_argument("--source", default="local", choices=["local", "hf"],
                   help="local=use ./data files; hf=stream big datasets (Colab)")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--segment_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4, help="per-step micro-batch (keep small on T4)")
    p.add_argument("--grad_accum", type=int, default=8, help="effective batch = batch_size * grad_accum")
    p.add_argument("--pretrain_steps", type=int, default=15000,
                   help="~500M tokens at bs8*ga4*1024. Scale up for a real model "
                        "(350M wants ~7B tokens -> ~210k steps).")
    p.add_argument("--finetune_steps", type=int, default=1500)
    p.add_argument("--pretrain_lr", type=float, default=3e-4,
                   help="3e-4 is stable for 350M; 6e-4 tends to spike early.")
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--no_grad_checkpoint", action="store_true",
                   help="disable gradient checkpointing (faster, but much more VRAM)")
    p.add_argument("--compile", action="store_true",
                   help="enable torch.compile (faster). OFF by default: Triton "
                        "currently fails to compile on Blackwell/sm_120 (RTX 50xx).")
    p.add_argument("--skip_pretrain", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    # T4 has no bf16; use fp16. A100/Ampere+ -> bf16 (more stable).
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[setup] device={device} amp={amp_dtype}")

    tok = ChatTokenizer()
    cfg = make_config(args.size, tok.vocab_size, args.seq_len, args.segment_size)
    model = RNNLM(cfg).to(device)
    
    if args.compile and device.type == "cuda":
        model = torch.compile(model)
        print("[setup] torch.compile enabled")
    else:
        print("[setup] torch.compile disabled (eager mode) — pass --compile to enable")

    model.grad_checkpoint = not args.no_grad_checkpoint
    print(f"[model] size={args.size} | {model.num_params()/1e6:.1f}M non-embedding params "
          f"| {sum(p.numel() for p in model.parameters())/1e6:.1f}M total "
          f"| grad_checkpoint={model.grad_checkpoint}")
    tok_per_step = args.batch_size * args.grad_accum * args.seq_len
    print(f"[budget] {tok_per_step:,} tokens/step | pretrain ~{tok_per_step*args.pretrain_steps/1e6:.0f}M tokens "
          f"| rule of thumb: 350M wants ~7B for full quality")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    pre_ckpt = CKPT_DIR / "pretrain.pt"
    sft_ckpt = CKPT_DIR / "finetune.pt"

    # ---------------- STAGE 1: PRE-TRAINING ---------------------------------
    if not args.skip_pretrain:
        print("\n=== STAGE 1: PRE-TRAINING ===")
        pre_data = PretrainData(tok, args.seq_len, args.source, device)
        run_stage(model, pre_data, stage="pretrain", steps=args.pretrain_steps,
                  batch_size=args.batch_size, grad_accum=args.grad_accum,
                  base_lr=args.pretrain_lr, warmup=max(100, args.pretrain_steps // 50),
                  device=device, amp_dtype=amp_dtype, grad_clip=1.0,
                  log_every=20, save_every=500, ckpt_path=pre_ckpt)
        export_weights(WEIGHTS_DIR / "model_pretrained.pt", model)
    elif pre_ckpt.exists():
        _raw_model(model).load_state_dict(torch.load(pre_ckpt, map_location=device)["model"])
        print("[setup] loaded pretrained weights, skipping stage 1")

    # ---------------- STAGE 2: CONVERSATIONAL FINE-TUNING -------------------
    print("\n=== STAGE 2: CONVERSATIONAL FINE-TUNING ===")
    records = load_chat_records(args.source)
    chat_data = ChatData(tok, records, args.seq_len, device)
    run_stage(model, chat_data, stage="finetune", steps=args.finetune_steps,
              batch_size=args.batch_size, grad_accum=args.grad_accum,
              base_lr=args.finetune_lr, warmup=max(50, args.finetune_steps // 20),
              device=device, amp_dtype=amp_dtype, grad_clip=1.0,
              log_every=20, save_every=500, ckpt_path=sft_ckpt)

    # ---------------- EXPORT FINAL WEIGHTS ---------------------------------
    final = WEIGHTS_DIR / "model_final.pt"
    export_weights(final, model)
    print(f"\n[done] Final conversational weights -> {final}")
    print("Download this file and place it in First_RNN_1B/weights/, then run "
          "`streamlit run inference.py`.")


if __name__ == "__main__":
    main()
