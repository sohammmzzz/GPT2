"""
data.py
=======
Everything that turns raw text on the internet into batches of integer tensors
the model can train on.  Broken into small functions so each stage is clear:

    download_corpus()  -> grabs + cleans public-domain text, writes corpus.txt
    CharTokenizer      -> maps characters <-> integer ids
    load_text()        -> reads corpus.txt
    encode_corpus()    -> text -> 1-D LongTensor of ids
    train_val_split()  -> split the id stream into train / val
    get_batch()        -> sample a (input, target) mini-batch for BPTT
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Tuple

import torch

from config import Config


# --------------------------------------------------------------------------- #
#  1. Downloading the raw corpus
# --------------------------------------------------------------------------- #

# Public-domain sources.  Project Gutenberg "plain text utf-8" mirrors give us
# clean, fluent modern-ish English prose; combining a few books yields a corpus
# big enough to learn real word/sentence structure but small enough to train
# quickly on a 6 GB laptop GPU.
_GUTENBERG_BOOKS = {
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "alice_in_wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
    "sherlock_holmes":     "https://www.gutenberg.org/files/1661/1661-0.txt",
    "a_tale_of_two_cities":"https://www.gutenberg.org/files/98/98-0.txt",
    "frankenstein":        "https://www.gutenberg.org/files/84/84-0.txt",
    "moby_dick":           "https://www.gutenberg.org/files/2701/2701-0.txt",
    "great_expectations":  "https://www.gutenberg.org/files/1400/1400-0.txt",
    "dracula":             "https://www.gutenberg.org/files/345/345-0.txt",
    "jane_eyre":           "https://www.gutenberg.org/files/1260/1260-0.txt",
    "huckleberry_finn":    "https://www.gutenberg.org/files/76/76-0.txt",
    "dorian_gray":         "https://www.gutenberg.org/files/174/174-0.txt",
    "war_of_the_worlds":   "https://www.gutenberg.org/files/36/36-0.txt",
}

_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def _http_get(url: str, timeout: int = 60) -> str:
    """Download a URL and return its text (utf-8, errors ignored)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove the legal header/footer Project Gutenberg wraps every book in, so the
    model only ever sees the actual prose.
    """
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                      text, flags=re.IGNORECASE | re.DOTALL)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                    text, flags=re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()] if not start else text[:end.start() - start.end()]
    return text


def _normalise(text: str) -> str:
    """Light cleanup: normalise newlines/whitespace, drop weird control chars."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip non-printable control characters except newline/tab
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
    return text.strip() + "\n"


def download_corpus(cfg: Config, force: bool = False) -> Path:
    """
    Ensure ``cfg.corpus_file`` exists.  Downloads + cleans the chosen dataset the
    first time, then just reuses the cached file.
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if cfg.corpus_file.exists() and not force:
        print(f"[data] using cached corpus: {cfg.corpus_file}")
        return cfg.corpus_file

    parts: list[str] = []
    if cfg.dataset == "gutenberg":
        for name, url in _GUTENBERG_BOOKS.items():
            try:
                print(f"[data] downloading {name} ...")
                raw = _http_get(url)
                parts.append(_normalise(_strip_gutenberg_boilerplate(raw)))
            except Exception as e:  # noqa: BLE001 - keep going if one book fails
                print(f"[data]   ! failed ({e}); skipping {name}")
    elif cfg.dataset == "shakespeare":
        print("[data] downloading tiny-shakespeare ...")
        parts.append(_normalise(_http_get(_SHAKESPEARE_URL)))
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}")

    if not parts:
        raise RuntimeError("No data could be downloaded - check your connection.")

    corpus = "\n\n".join(parts)
    cfg.corpus_file.write_text(corpus, encoding="utf-8")
    print(f"[data] wrote {len(corpus):,} characters -> {cfg.corpus_file}")
    return cfg.corpus_file


# --------------------------------------------------------------------------- #
#  2. Character tokenizer
# --------------------------------------------------------------------------- #

class CharTokenizer:
    """
    The simplest possible tokenizer: every distinct character is one token.

    Holds two lookup tables, ``stoi`` (string->int) and ``itos`` (int->string),
    built from whatever text you fit it on.
    """

    def __init__(self, chars: list[str]):
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from the sorted set of characters in ``text``."""
        chars = sorted(set(text))
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # -- persistence so generation can reuse the exact same vocab -----------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        itos = json.loads(Path(path).read_text(encoding="utf-8"))["itos"]
        chars = [itos[str(i)] for i in range(len(itos))]
        return cls(chars)


# --------------------------------------------------------------------------- #
#  3. Text -> tensors
# --------------------------------------------------------------------------- #

def load_text(cfg: Config) -> str:
    return cfg.corpus_file.read_text(encoding="utf-8")


def encode_corpus(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    """Whole corpus as one long 1-D tensor of token ids (kept on CPU)."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def train_val_split(data: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use the final ``val_fraction`` of the stream as validation."""
    n_val = int(len(data) * cfg.val_fraction)
    n_train = len(data) - n_val
    return data[:n_train], data[n_train:]


def get_batch(data: torch.Tensor, cfg: Config, device: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a random mini-batch for truncated BPTT.

    For each of ``batch_size`` random start positions we take ``seq_len`` chars
    as the input ``x`` and the same window shifted right by one as the target
    ``y`` (next-character prediction).  Shapes: (batch_size, seq_len).
    """
    max_start = len(data) - cfg.seq_len - 1
    starts = torch.randint(0, max_start, (cfg.batch_size,))
    x = torch.stack([data[s: s + cfg.seq_len] for s in starts])
    y = torch.stack([data[s + 1: s + 1 + cfg.seq_len] for s in starts])
    # pin + non_blocking copy keeps the GPU fed without stalling
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def prepare_data(cfg: Config, device: torch.device):
    """
    Convenience one-shot: download (if needed), fit tokenizer, encode, split.

    Returns ``(train_data, val_data, tokenizer)`` and updates ``cfg.vocab_size``.
    """
    download_corpus(cfg)
    text = load_text(cfg)
    tokenizer = CharTokenizer.fit(text)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"[data] corpus chars: {len(text):,} | vocab size: {cfg.vocab_size}")
    data = encode_corpus(text, tokenizer)
    train_data, val_data = train_val_split(data, cfg)
    print(f"[data] train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")
    return train_data, val_data, tokenizer


if __name__ == "__main__":
    # `python data.py` just downloads + reports stats, handy for a first run.
    from config import get_config
    cfg = get_config()
    if len(sys.argv) > 1:
        cfg.dataset = sys.argv[1]
        cfg.corpus_file = cfg.data_dir / "corpus.txt"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_data(cfg, dev)
