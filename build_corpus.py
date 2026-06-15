"""
build_corpus.py — Web-sourced data pipeline (NO fabricated data)
================================================================

This script assembles REAL training data from public web sources into two files
that `train.py` can consume directly:

    data/corpus.txt   -> plain-text pre-training corpus (general English)
    data/chat.jsonl   -> conversational SFT data, one JSON record per line:
                         {"messages": [{"role": "...", "content": "..."}, ...]}

It is intentionally dependency-light (only `requests`, already in your venv) so
you can build a real, inspectable sample LOCALLY. On Colab, `train.py` can pull
*much* larger streaming datasets directly (FineWeb-Edu, OpenAssistant, etc.) —
see `--source hf` there. This file gives you a tangible, offline corpus and
proves the formatting pipeline end-to-end.

Sources used here (all public / open-licensed):
  * Project Gutenberg  -> public-domain books (pre-training text)
  * databricks-dolly-15k (Apache-2.0) -> instruction/response pairs (SFT)
  * OpenAssistant oasst1 (Apache-2.0) -> multi-turn assistant dialogues (SFT)

Run:
    python build_corpus.py                 # default sample
    python build_corpus.py --max_books 12  # bigger pre-training corpus
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# --------------------------------------------------------------------------- #
#  HTTP helper
# --------------------------------------------------------------------------- #

def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (corpus-builder)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_text(url: str, timeout: int = 120) -> str:
    return http_get(url, timeout).decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
#  1. Pre-training text  (Project Gutenberg public-domain books)
# --------------------------------------------------------------------------- #

GUTENBERG_BOOKS = {
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
    "ulysses":             "https://www.gutenberg.org/files/4300/4300-0.txt",
    "metamorphosis":       "https://www.gutenberg.org/files/5200/5200-0.txt",
    "count_monte_cristo":  "https://www.gutenberg.org/files/1184/1184-0.txt",
}


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                      text, flags=re.IGNORECASE | re.DOTALL)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                    text, flags=re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    if end:
        idx = text.find("*** END OF")
        if idx > 0:
            text = text[:idx]
    return text


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(ch for ch in text if ch in "\n\t" or ch >= " ")
    return text.strip() + "\n"


def build_pretrain_corpus(max_books: int) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "corpus.txt"
    parts: list[str] = []
    for i, (name, url) in enumerate(GUTENBERG_BOOKS.items()):
        if i >= max_books:
            break
        try:
            print(f"[corpus] downloading {name} ...")
            parts.append(normalise(strip_gutenberg(http_get_text(url))))
        except Exception as e:  # noqa: BLE001
            print(f"[corpus]   ! failed ({e}); skipping {name}")
    if not parts:
        raise RuntimeError("No pre-training text downloaded — check your connection.")
    text = "\n\n".join(parts)
    out.write_text(text, encoding="utf-8")
    print(f"[corpus] wrote {len(text):,} chars -> {out}")
    return out


# --------------------------------------------------------------------------- #
#  2. Conversational SFT data
# --------------------------------------------------------------------------- #

DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
             "resolve/main/databricks-dolly-15k.jsonl")

# OpenAssistant trees (ready-made conversation trees, gzip-jsonl)
OASST_TREES_URL = ("https://huggingface.co/datasets/OpenAssistant/oasst1/"
                   "resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz")


def dolly_to_messages(rec: dict) -> list[dict] | None:
    """Convert one dolly record into a chat message list."""
    instruction = (rec.get("instruction") or "").strip()
    context = (rec.get("context") or "").strip()
    response = (rec.get("response") or "").strip()
    if not instruction or not response:
        return None
    user = instruction if not context else f"{instruction}\n\n{context}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]


def fetch_dolly(records: list[dict]) -> None:
    try:
        print("[chat] downloading databricks-dolly-15k ...")
        raw = http_get_text(DOLLY_URL)
    except Exception as e:  # noqa: BLE001
        print(f"[chat]   ! dolly failed ({e}); skipping")
        return
    n = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msgs = dolly_to_messages(json.loads(line))
        except Exception:
            continue
        if msgs:
            records.append({"messages": msgs})
            n += 1
    print(f"[chat]   + {n:,} dolly conversations")


def _walk_oasst_tree(node: dict, prefix: list[dict], out: list[list[dict]]) -> None:
    """Depth-first walk of an oasst conversation tree, emitting prompter/assistant
    message chains that END on an assistant turn (valid SFT targets)."""
    role = "user" if node.get("role") == "prompter" else "assistant"
    text = (node.get("text") or "").strip()
    chain = prefix + [{"role": role, "content": text}] if text else prefix
    replies = node.get("replies") or []
    if role == "assistant" and len(chain) >= 2:
        out.append(chain)
    for child in replies:
        _walk_oasst_tree(child, chain, out)


def fetch_oasst(records: list[dict], max_threads: int) -> None:
    try:
        print("[chat] downloading OpenAssistant oasst1 trees ...")
        blob = http_get(OASST_TREES_URL)
        text = gzip.GzipFile(fileobj=io.BytesIO(blob)).read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        print(f"[chat]   ! oasst failed ({e}); skipping")
        return
    n = 0
    for line in text.splitlines():
        if n >= max_threads:
            break
        line = line.strip()
        if not line:
            continue
        try:
            tree = json.loads(line)
        except Exception:
            continue
        prompt = tree.get("prompt")
        if not prompt or (tree.get("prompt", {}).get("lang") not in (None, "en")):
            continue
        chains: list[list[dict]] = []
        _walk_oasst_tree(prompt, [], chains)
        # keep the single highest-quality (shortest valid) chain per tree to avoid blow-up
        for chain in chains[:2]:
            records.append({"messages": chain})
            n += 1
    print(f"[chat]   + {n:,} oasst conversations")


def build_chat_data(max_oasst: int) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "chat.jsonl"
    records: list[dict] = []
    fetch_dolly(records)
    fetch_oasst(records, max_oasst)
    if not records:
        raise RuntimeError("No SFT data downloaded — check your connection.")
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[chat] wrote {len(records):,} conversations -> {out}")
    return out


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Build pretraining + SFT data from the web")
    p.add_argument("--max_books", type=int, default=15, help="Gutenberg books for corpus.txt")
    p.add_argument("--max_oasst", type=int, default=8000, help="max oasst conversations")
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument("--skip_chat", action="store_true")
    args = p.parse_args()

    if not args.skip_pretrain:
        build_pretrain_corpus(args.max_books)
    if not args.skip_chat:
        build_chat_data(args.max_oasst)
    print("\n[done] Local data ready in ./data/. "
          "For large-scale pretraining, use train.py --source hf on Colab.")


if __name__ == "__main__":
    main()
