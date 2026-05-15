import os, uuid, json, math, time, tempfile
import torch
from TOKENIZATION_AND_EMBEDDING import create_dataloader_v1, raw_text
from TRANSFORMER import GPTModel


# ── Output location (on Z: drive — outside OneDrive, no sync conflicts) ───────
OUTPUT_DIR = r"Z:\MY_MODEL"


# ── Training Configuration — edit these to experiment ─────────────────────────
CONFIG = {
    # Model architecture
    "vocab_size":     50257,
    "context_length": 128,
    "d_model":        256,
    "num_heads":      8,       # d_head = 32
    "num_layers":     6,
    "dropout":        0.1,
    # Data  — stride < context_length creates overlapping windows (more samples)
    "batch_size":     2,
    "stride":         32,
    # Optimiser
    "epochs":         10,
    "lr":             3e-4,
    "weight_decay":   0.1,
    "betas":          [0.9, 0.95],
    "grad_clip":      1.0,
    # LR schedule (linear warmup → cosine decay)
    "warmup_steps":   50,
    # GPU memory
    "amp":            True,    # AMP mixed precision — halves activation memory on CUDA
    # Logging
    "log_interval":   5,
}


# ── Run Directory (under OUTPUT_DIR) ─────────────────────────────────────────
def get_run_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("run_")]
    run_num  = len(existing) + 1
    uid      = uuid.uuid4().hex[:8]
    path     = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_{uid}")
    os.makedirs(path)
    return path


# ── LR Schedule: linear warmup + cosine decay ────────────────────────────────
def build_lr_lambda(warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ── Atomic Save: temp file lives on the same drive as the destination ─────────
def _atomic_save(obj, path):
    dest_dir = os.path.dirname(os.path.abspath(path))
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt.tmp", dir=dest_dir)
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ── Main Training Loop ────────────────────────────────────────────────────────
def train(cfg=CONFIG):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["amp"] and device.type == "cuda"

    print(f"\nDevice     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"CUDA       : {torch.version.cuda}")
        print(f"VRAM total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"AMP        : {use_amp}")

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=cfg["batch_size"],
        max_length=cfg["context_length"],
        stride=cfg["stride"],
        shuffle=True,
        drop_last=True,
    )

    if len(dataloader) == 0:
        raise RuntimeError(
            f"Dataloader is empty — text too short for these settings.\n"
            f"  batch_size={cfg['batch_size']}, context_length={cfg['context_length']}, stride={cfg['stride']}\n"
            f"  Reduce batch_size or context_length."
        )

    print(f"Samples    : {len(dataloader.dataset)}")
    print(f"Batches/ep : {len(dataloader)}\n")

    model = GPTModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        max_seq_len=cfg["context_length"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {params:,}")
    if device.type == "cuda":
        print(f"Model VRAM : {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["betas"]),
    )

    total_steps = len(dataloader) * cfg["epochs"]
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(cfg["warmup_steps"], total_steps)
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = get_run_dir()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saving to  : {run_dir}\n")

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (inputs, targets) in enumerate(dataloader, 1):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(inputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if step % cfg["log_interval"] == 0:
                avg  = epoch_loss / step
                lr   = scheduler.get_last_lr()[0]
                vram = torch.cuda.memory_allocated() / 1e6 if device.type == "cuda" else 0
                print(f"  [{epoch:02d}] step {step:3d}/{len(dataloader)} | loss={avg:.4f} | lr={lr:.2e} | VRAM={vram:.0f}MB")

        avg_loss = epoch_loss / len(dataloader)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | loss={avg_loss:.4f} | {elapsed:.1f}s")

        _atomic_save(
            {"epoch": epoch, "model": model.state_dict(), "config": cfg},
            os.path.join(run_dir, f"checkpoint_epoch_{epoch:03d}.pt"),
        )

    _atomic_save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))
    print(f"\nTraining complete: {run_dir}\\model_final.pt")
    return run_dir


if __name__ == "__main__":
    train()
