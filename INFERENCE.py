import os, json
import torch
import tiktoken
from TRANSFORMER import GPTModel


# ── Model weights location ────────────────────────────────────────────────────
OUTPUT_DIR = r"Z:\MY_MODEL"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Loading ─────────────────────────────────────────────────────────────
def list_runs():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory not found: {OUTPUT_DIR}")
        return []
    return sorted(d for d in os.listdir(OUTPUT_DIR) if d.startswith("run_"))


def load_model(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    model = GPTModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
        max_seq_len=cfg["context_length"],
    ).to(DEVICE)

    final = os.path.join(run_dir, "model_final.pt")
    if os.path.exists(final):
        model.load_state_dict(torch.load(final, map_location=DEVICE, weights_only=True))
        print(f"Loaded  : model_final.pt")
    else:
        ckpts = sorted(f for f in os.listdir(run_dir) if f.startswith("checkpoint"))
        if not ckpts:
            raise FileNotFoundError(f"No weights found in {run_dir}")
        ckpt = torch.load(os.path.join(run_dir, ckpts[-1]), map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded  : {ckpts[-1]}")

    model.eval()
    return model, cfg


# ── Text Generation (top-k sampling with temperature) ────────────────────────
@torch.no_grad()
def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=50, context_length=256):
    tokenizer = tiktoken.get_encoding("gpt2")
    ids       = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(DEVICE)

    for _ in range(max_new_tokens):
        ids_cond  = ids[:, -context_length:]
        logits, _ = model(ids_cond)
        logits    = logits[:, -1, :] / temperature

        if top_k:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, [-1]]] = float("-inf")

        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())


if __name__ == "__main__":
    print(f"\nDevice     : {DEVICE}")
    print(f"Model store: {OUTPUT_DIR}\n")

    runs = list_runs()
    if not runs:
        print("No trained models found.")
        raise SystemExit

    print("Available runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r}")

    def _int_input(msg, default=None):
        while True:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            try:
                return int(raw)
            except ValueError:
                hint = f" (or leave blank for {default})" if default is not None else ""
                print(f"  Please enter a whole number{hint}.")

    def _float_input(msg, default=None):
        while True:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            try:
                return float(raw)
            except ValueError:
                hint = f" (or leave blank for {default})" if default is not None else ""
                print(f"  Please enter a decimal number{hint}.")

    while True:
        idx = _int_input("\nSelect run index: ")
        if 0 <= idx < len(runs):
            break
        print(f"  Must be between 0 and {len(runs) - 1}.")

    run_dir = os.path.join(OUTPUT_DIR, runs[idx])
    model, cfg = load_model(run_dir)
    print(f"Source  : {run_dir}")
    print(f"Config  : d_model={cfg['d_model']}, layers={cfg['num_layers']}, heads={cfg['num_heads']}\n")

    prompt = input("Prompt         : ").strip() or "The"
    n_tok  = _int_input("Max new tokens [100] : ", default=100)
    temp   = _float_input("Temperature    [1.0] : ", default=1.0)
    top_k  = _int_input("Top-k          [50]  : ", default=50)

    output = generate(
        model, prompt,
        max_new_tokens=n_tok,
        temperature=temp,
        top_k=top_k,
        context_length=cfg["context_length"],
    )
    print(f"\n{'='*60}\n{output}\n{'='*60}")
