import torch
import torch.nn as nn
from TOKENIZATION_AND_EMBEDDING import VOCAB_SIZE, EMBED_DIM
from blocks.MASKED_ATTENTION import MaskedMultiHeadAttention
from blocks.FEED_FORWARD import FeedForward


# ── Transformer Block (Pre-LayerNorm GPT-style decoder block) ────────────────
class TransformerBlock(nn.Module):
    """LN → MaskedMHA → residual → LN → FFN → residual"""

    def __init__(self, d_model, num_heads, dropout, max_seq_len):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MaskedMultiHeadAttention(d_model, num_heads, dropout, max_seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ── GPT Model (decoder-only transformer for next-token prediction) ────────────
class GPTModel(nn.Module):
    """Token emb + positional emb → N TransformerBlocks → LN → LM head."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=EMBED_DIM, num_heads=8,
                 num_layers=6, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, S  = input_ids.shape
        pos   = torch.arange(S, device=input_ids.device)
        x     = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_final(x))   # (B, S, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss


if __name__ == "__main__":
    print("\n____ GPT Transformer Demo ____\n")
    model        = GPTModel(d_model=256, num_heads=8, num_layers=6, max_seq_len=256)
    ids          = torch.randint(0, VOCAB_SIZE, (2, 64))
    tgts         = torch.randint(0, VOCAB_SIZE, (2, 64))
    logits, loss = model(ids, tgts)
    params       = sum(p.numel() for p in model.parameters())
    print(f"Logits     : {logits.shape}")
    print(f"Loss       : {loss.item():.4f}")
    print(f"Parameters : {params:,}")
