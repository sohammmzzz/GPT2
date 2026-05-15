import torch
import torch.nn as nn


# ── Position-wise Feed-Forward Network ──────────────────────────────────────
class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation. Inner dim = d_model × expansion (GPT uses 4)."""

    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("\n____ Feed-Forward Demo ____\n")
    ffn = FeedForward(d_model=256, expansion=4)
    x   = torch.randn(2, 10, 256)
    out = ffn(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
