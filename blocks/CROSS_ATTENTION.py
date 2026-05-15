import torch
import torch.nn as nn


# ── Cross-Attention (Encoder → Decoder) ─────────────────────────────────────
class CrossAttention(nn.Module):
    """Cross-attention: queries from decoder (x), keys and values from encoder output (context)."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
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

    def forward(self, x, context):
        """
        x       : (B, S_dec, d_model)  — decoder sequence
        context : (B, S_enc, d_model)  — encoder output (can differ in length)
        """
        B, S, _ = x.shape

        Q = self._split_heads(self.W_query(x))        # decoder queries
        K = self._split_heads(self.W_key(context))    # encoder keys
        V = self._split_heads(self.W_value(context))  # encoder values

        scale        = self.d_head ** 0.5
        attn_scores  = (Q @ K.transpose(-2, -1)) / scale   # (B, H, S_dec, S_enc)
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))

        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_out(out), attn_weights


if __name__ == "__main__":
    print("\n____ Cross-Attention Demo ____\n")
    ca      = CrossAttention(d_model=256, num_heads=8)
    x       = torch.randn(2, 10, 256)    # decoder: 10 tokens
    context = torch.randn(2, 20, 256)   # encoder: 20 tokens (different length ok)
    out, w  = ca(x, context)
    print(f"Decoder input   : {x.shape}")
    print(f"Encoder context : {context.shape}")
    print(f"Output          : {out.shape}")
    print(f"Attn weights    : {w.shape}")  # (B, H, S_dec, S_enc)
