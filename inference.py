"""
inference.py — Streamlit chat UI for the Memory-Caching conversational RNN
==========================================================================

Loads the weights you trained on Colab (./weights/model_final.pt) and lets you
chat with the model on your local GPU. The model is a recurrent network with
Memory Caching, so it runs comfortably even on a 6 GB GPU in fp16.

Run:
    pip install streamlit torch tiktoken
    streamlit run inference.py

The architecture is imported from train.py (single source of truth), so the
model you infer is byte-for-byte the one you trained.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import streamlit as st

from train import (
    RNNLM, ModelConfig, ChatTokenizer,
    SYSTEM_TAG, USER_TAG, ASSISTANT_TAG, DEFAULT_SYSTEM, WEIGHTS_DIR,
)

st.set_page_config(page_title="Memory-Caching RNN Chat", page_icon="🧠", layout="centered")


# --------------------------------------------------------------------------- #
#  Model loading (cached so we only load once per session)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=True)
def load_model(weights_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = WEIGHTS_DIR / weights_name
    if not path.exists():
        return None, None, device, None
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = RNNLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # fp16 on GPU for speed + low memory; fp32 on CPU
    if device.type == "cuda":
        model = model.half()
    tok = ChatTokenizer()
    n_params = sum(p.numel() for p in model.parameters())
    return model, tok, device, n_params


# --------------------------------------------------------------------------- #
#  Streaming generation (recompute-based; Memory Caching fully active)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def stream_reply(model, tok, device, history, system, *,
                 max_new_tokens, temperature, top_k, top_p, repetition_penalty,
                 context_window=1024):
    """Yield the assistant reply text incrementally for a typing effect."""
    prompt_ids = tok.build_inference_prompt(history, system=system)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    prev_text = ""
    for _ in range(max_new_tokens):
        ctx = idx[:, -context_window:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].float()
        # discourage repetition
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
            sp = sp / sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
        else:
            nxt = torch.multinomial(probs, 1)
        nid = int(nxt)
        if nid == tok.eot:
            break
        generated.append(nid)
        idx = torch.cat([idx, nxt], dim=1)
        # decode the whole list and emit only the new suffix (BPE-safe)
        text = tok.decode(generated)
        if text != prev_text:
            yield text[len(prev_text):]
            prev_text = text


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #

st.title("🧠 Memory-Caching RNN")
st.caption("A recurrent language model with Memory Caching (Behrouz et al., 2026) — "
           "a growing-memory RNN that aims for transformer-level recall, running on your GPU.")

with st.sidebar:
    st.header("⚙️ Settings")
    weights_choice = st.selectbox(
        "Weights file",
        options=["model_final.pt", "model_pretrained.pt"],
        help="model_final.pt = after conversational fine-tuning (recommended).",
    )
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM, height=80)
    st.divider()
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    top_k = st.slider("Top-k", 0, 200, 50, 5)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)
    max_new_tokens = st.slider("Max new tokens", 16, 768, 256, 16)
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

model, tok, device, n_params = load_model(weights_choice)

if model is None:
    st.warning(
        f"**No weights found at `weights/{weights_choice}`.**\n\n"
        "Train the model first:\n"
        "1. Copy `train.py` to a Colab notebook (free T4 GPU).\n"
        "2. Run `!python train.py --size 350m --source hf`.\n"
        "3. Download the produced `weights/model_final.pt` into this folder's "
        "`weights/` directory.\n\n"
        "Then reload this page."
    )
    st.stop()

st.sidebar.success(f"Loaded **{weights_choice}**\n\n"
                   f"{n_params/1e6:.0f}M params · {device.type.upper()}")

if "history" not in st.session_state:
    st.session_state.history = []

# replay history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# new user turn
if user_input := st.chat_input("Message the model…"):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        t0 = time.time()
        reply = st.write_stream(stream_reply(
            model, tok, device, st.session_state.history, system_prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            context_window=model.cfg.max_seq_len,
        ))
        dt = time.time() - t0
        st.caption(f"generated in {dt:.1f}s")
    st.session_state.history.append({"role": "assistant", "content": reply})
