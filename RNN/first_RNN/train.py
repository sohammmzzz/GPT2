"""
train.py
========
The training loop, split into named helper functions so each responsibility is
obvious:

    set_seed()           - reproducibility
    lr_at_step()         - warmup + cosine learning-rate schedule
    evaluate()           - average loss over the validation split
    train_step()         - one forward/backward/optimiser step (with AMP)
    save_checkpoint()    - persist the best model + its config + tokenizer
    train()              - glue that runs the whole thing

Loss is plain cross-entropy on next-character prediction.  We also report
"perplexity" (= exp(loss)) which is the more human-readable quality metric for a
language model: lower is better, and it roughly means "how many characters the
model is choosing between on average".
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from config import Config, get_config
from data import prepare_data
from model import build_model
from generate import generate


# --------------------------------------------------------------------------- #
#  Reproducibility & schedule
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at_step(step: int, total_steps: int, cfg: Config) -> float:
    """Linear warmup for ``warmup_steps`` then cosine decay down to a floor."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, val_data, cfg, device, n_batches: int = 50) -> float:
    """Mean cross-entropy over a handful of random validation batches."""
    from data import get_batch
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_batches):
        x, y = get_batch(val_data, cfg, device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# --------------------------------------------------------------------------- #
#  A single optimisation step
# --------------------------------------------------------------------------- #

def train_step(model, x, y, optimizer, criterion, scaler, cfg) -> float:
    """One forward + backward + update.  Returns the scalar loss."""
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

    if cfg.use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    return loss.item()


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(model, tokenizer, cfg, val_loss, step) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.ckpt_dir / "tokenizer.json")
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "val_loss": val_loss,
            "step": step,
        },
        cfg.ckpt_file,
    )
    print(f"[ckpt] saved best (val_loss={val_loss:.4f}) -> {cfg.ckpt_file}")


# --------------------------------------------------------------------------- #
#  Main training routine
# --------------------------------------------------------------------------- #

def train(cfg: Config) -> None:
    from data import get_batch

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # speed knobs that are safe + helpful on the RTX 4050
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[train] device: {device}")

    # --- data + model ----------------------------------------------------- #
    train_data, val_data, tokenizer = prepare_data(cfg, device)
    model = build_model(cfg, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    # One "epoch" = enough random batches to have seen ~the whole train stream.
    tokens_per_batch = cfg.batch_size * cfg.seq_len
    steps_per_epoch = max(1, len(train_data) // tokens_per_batch)
    total_steps = steps_per_epoch * cfg.epochs
    print(f"[train] steps/epoch: {steps_per_epoch} | total steps: {total_steps}")

    best_val = float("inf")
    running = []
    t0 = time.time()
    step = 0

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        for _ in range(steps_per_epoch):
            # set the scheduled learning rate for this step
            lr = lr_at_step(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, cfg, device)
            loss = train_step(model, x, y, optimizer, criterion, scaler, cfg)
            running.append(loss)
            step += 1

            # ---- periodic logging ---------------------------------------- #
            if step % cfg.log_every == 0:
                avg = float(np.mean(running[-cfg.log_every:]))
                speed = step * tokens_per_batch / (time.time() - t0)
                print(f"epoch {epoch:>2} | step {step:>6}/{total_steps} "
                      f"| lr {lr:.2e} | loss {avg:.4f} | ppl {math.exp(avg):7.2f} "
                      f"| {speed/1e3:.0f}k tok/s")

            # ---- periodic validation + checkpoint ------------------------ #
            if step % cfg.eval_every == 0:
                val_loss = evaluate(model, val_data, cfg, device)
                print(f"  -> val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.2f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, cfg, val_loss, step)

            # ---- periodic text sample ------------------------------------ #
            if step % cfg.sample_every == 0:
                sample = generate(model, tokenizer, device, prompt="The ",
                                  length=240, temperature=cfg.temperature,
                                  top_k=cfg.top_k)
                print("  --- sample ---\n" + _indent(sample) + "\n  --------------")
                model.train()

    # Final evaluation + make sure we have a checkpoint even if eval_every never hit.
    val_loss = evaluate(model, val_data, cfg, device)
    print(f"[train] done. final val loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
    if val_loss < best_val:
        save_checkpoint(model, tokenizer, cfg, val_loss, step)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a character-level RNN LM")
    parser.add_argument("--dataset", default=None, help="gutenberg | shakespeare")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--cell_type", default=None, help="lstm | gru | rnn")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = get_config(**overrides)
    train(cfg)
