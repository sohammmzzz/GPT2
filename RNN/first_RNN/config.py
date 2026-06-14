"""
config.py
=========
Central place for every hyper-parameter and path used by the project.

Keeping all the knobs in one small dataclass means the rest of the code never
hard-codes a magic number, and you can see the whole "shape" of the experiment
at a glance.  The defaults below are tuned for a 6 GB laptop GPU
(NVIDIA RTX 4050 Laptop).
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

# Project lives in .../RNN/first_RNN
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"


@dataclass
class Config:
    # ----------------------------------------------------------------- data
    # Which corpus to download/use. Options handled in data.py:
    #   "gutenberg"  -> a few public-domain novels (modern, fluent English)
    #   "shakespeare"-> the classic karpathy tiny-shakespeare file
    dataset: str = "gutenberg"
    data_dir: Path = DATA_DIR
    corpus_file: Path = DATA_DIR / "corpus.txt"   # the cleaned, concatenated text
    val_fraction: float = 0.1                      # last 10% of text held out for validation

    # ------------------------------------------------------------- tokenizer
    # Character-level: the "vocabulary" is just the set of unique characters.
    # vocab_size is filled in at runtime once we have seen the data.
    vocab_size: int = 0

    # ----------------------------------------------------------------- model
    cell_type: str = "lstm"      # "lstm" | "gru" | "rnn"  (lstm = best quality)
    embedding_dim: int = 256     # size of the learned per-character vector
    hidden_size: int = 512       # width of the recurrent hidden state
    num_layers: int = 3          # stacked recurrent layers
    dropout: float = 0.2         # dropout between layers (regularisation)
    tie_weights: bool = False    # only valid if embedding_dim == hidden_size

    # -------------------------------------------------------------- training
    seq_len: int = 256           # truncated-backprop-through-time window
    batch_size: int = 64         # sequences per step (fits comfortably in 6 GB)
    epochs: int = 12
    lr: float = 3e-3             # Adam learning rate
    weight_decay: float = 1e-5
    grad_clip: float = 5.0       # gradient norm clipping (RNNs explode without this)
    warmup_steps: int = 200      # linear LR war-up steps
    min_lr_ratio: float = 0.1    # cosine decay floor = lr * this
    use_amp: bool = True         # mixed precision -> faster + less VRAM on the 4050

    # ----------------------------------------------------------- bookkeeping
    seed: int = 1337
    log_every: int = 50          # steps between train-loss prints
    eval_every: int = 500        # steps between validation passes
    sample_every: int = 500      # steps between generating a text sample
    ckpt_dir: Path = CKPT_DIR
    ckpt_file: Path = CKPT_DIR / "best.pt"

    # ----------------------------------------------------------- generation
    gen_length: int = 500        # characters to generate when sampling
    temperature: float = 0.8     # <1 = safer/repetitive, >1 = more creative
    top_k: int = 40              # keep only the k most likely chars each step (0 = off)

    def to_dict(self) -> dict:
        """JSON-friendly view (Paths -> str) used when saving checkpoints."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def get_config(**overrides) -> Config:
    """Return a Config, optionally overriding any field by keyword."""
    cfg = Config()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config field: {key!r}")
        setattr(cfg, key, value)
    return cfg
