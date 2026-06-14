"""
model.py
========
The neural network itself, kept deliberately small and readable.

A character-level recurrent language model has exactly three pieces:

    1. Embedding      - turn each character id into a dense vector.
    2. Recurrent core - read the vectors one step at a time, carrying a hidden
                        state that summarises everything seen so far.
                        We expose LSTM / GRU / vanilla-RNN via one flag.
    3. Output head    - project the hidden state to a score for every possible
                        next character (the vocabulary).

The forward pass returns raw logits; turning them into a loss or into sampled
text is the caller's job (see train.py / generate.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


# Map the friendly name in the config to the actual torch module.
_RNN_CELLS = {
    "lstm": nn.LSTM,
    "gru": nn.GRU,
    "rnn": nn.RNN,
}


class CharRNN(nn.Module):
    """A stacked RNN/LSTM/GRU language model over a character vocabulary."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.cell_type = cfg.cell_type.lower()
        if self.cell_type not in _RNN_CELLS:
            raise ValueError(f"cell_type must be one of {list(_RNN_CELLS)}")

        # 1. Embedding table: (vocab_size, embedding_dim)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        # 2. Recurrent stack. batch_first=True -> tensors are (batch, time, feat)
        rnn_cls = _RNN_CELLS[self.cell_type]
        self.rnn = rnn_cls(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.drop = nn.Dropout(cfg.dropout)

        # 3. Output projection: hidden_size -> vocab_size
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

        # Optional weight tying (shares embedding & output matrices). Only makes
        # dimensional sense when embedding_dim == hidden_size.
        if cfg.tie_weights:
            if cfg.embedding_dim != cfg.hidden_size:
                raise ValueError("tie_weights requires embedding_dim == hidden_size")
            self.head.weight = self.embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small, sensible initialisation for stable early training."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x      : LongTensor (batch, time) of character ids.
        hidden : previous recurrent state, or None to start fresh.

        Returns (logits, hidden) where logits is (batch, time, vocab_size).
        """
        emb = self.drop(self.embedding(x))         # (B, T, E)
        out, hidden = self.rnn(emb, hidden)        # (B, T, H)
        out = self.drop(out)
        logits = self.head(out)                    # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Create a zero hidden state (LSTM needs a (h, c) tuple)."""
        h = torch.zeros(self.cfg.num_layers, batch_size,
                        self.cfg.hidden_size, device=device)
        if self.cell_type == "lstm":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: Config, device: torch.device) -> CharRNN:
    """Factory: construct the model, move it to the device, report its size."""
    model = CharRNN(cfg).to(device)
    print(f"[model] {cfg.cell_type.upper()} | "
          f"{cfg.num_layers} layers x {cfg.hidden_size} hidden | "
          f"{model.num_parameters()/1e6:.2f}M params")
    return model
