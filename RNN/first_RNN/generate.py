"""
generate.py
===========
Sampling text from a trained CharRNN, one character at a time.

The loop is the mirror image of training: feed the model a context, read off the
probability distribution for the *next* character, pick one, append it, repeat.
Two standard knobs shape the output:

    temperature - rescales logits.  <1 sharpens (safer, more repetitive),
                  >1 flattens (more surprising, more typos).
    top_k       - before sampling, zero out everything except the k most likely
                  characters.  Cheap way to kill obvious garbage.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config, get_config
from data import CharTokenizer
from model import CharRNN, build_model


@torch.no_grad()
def generate(model: CharRNN,
             tokenizer: CharTokenizer,
             device: torch.device,
             prompt: str = "\n",
             length: int = 500,
             temperature: float = 0.8,
             top_k: int = 40) -> str:
    """Autoregressively generate ``length`` characters starting from ``prompt``."""
    model.eval()

    # Warm the hidden state up on the prompt so generation continues from it.
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    hidden = model.init_hidden(batch_size=1, device=device)
    logits, hidden = model(context, hidden)
    logits = logits[:, -1, :]  # last time step only

    generated = list(prompt)
    for _ in range(length):
        logits = logits / max(temperature, 1e-6)

        if top_k and top_k < logits.size(-1):
            kth = torch.topk(logits, top_k).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        generated.append(tokenizer.decode(next_id[0]))

        logits, hidden = model(next_id, hidden)
        logits = logits[:, -1, :]

    return "".join(generated)


def load_for_generation(cfg: Config, device: torch.device):
    """Rebuild the model + tokenizer from a saved checkpoint."""
    ckpt = torch.load(cfg.ckpt_file, map_location=device)
    saved = ckpt["config"]
    # restore the architecture exactly as it was trained
    for key in ("cell_type", "embedding_dim", "hidden_size", "num_layers",
                "dropout", "tie_weights", "vocab_size"):
        setattr(cfg, key, saved[key])
    tokenizer = CharTokenizer.load(cfg.ckpt_dir / "tokenizer.json")
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample text from a trained CharRNN")
    parser.add_argument("--prompt", default="\n", help="seed text")
    parser.add_argument("--length", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_for_generation(cfg, device)

    text = generate(model, tokenizer, device,
                    prompt=args.prompt, length=args.length,
                    temperature=args.temperature, top_k=args.top_k)
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)
