import torch
import torch.nn as nn


# ── Multi-Head Self-Attention ────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with output projection."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key   = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.W_out   = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

    def forward(self, x):
        B, S, _ = x.shape

        Q = self._split_heads(self.W_query(x))   # (B, H, S, d_head)
        K = self._split_heads(self.W_key(x))
        V = self._split_heads(self.W_value(x))

        scale        = self.d_head ** 0.5
        attn_scores  = (Q @ K.transpose(-2, -1)) / scale
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))

        context = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_out(context), attn_weights


if __name__ == "__main__":
    print("\n____ Multi-Head Attention Demo ____\n")
    mha     = MultiHeadAttention(d_model=256, num_heads=8)
    x       = torch.randn(2, 10, 256)
    out, w  = mha(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Weights: {w.shape}")
