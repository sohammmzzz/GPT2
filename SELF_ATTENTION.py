import os
import torch
import torch.nn as nn
from TOKENIZATION_AND_EMBEDDING import create_dataloader_v1, raw_text, VOCAB_SIZE, EMBED_DIM


CONTEXT_LENGTH = 4   # sequence length (must match max_length in dataloader)
D_IN           = EMBED_DIM   # 256 — input dim from embedding layer
D_OUT          = EMBED_DIM   # 256 — output dim of each attention head


# ── Embedding layers (trained alongside QKV) ────────────────────────────────
token_emb_layer = nn.Embedding(VOCAB_SIZE, D_IN)
pos_emb_layer   = nn.Embedding(CONTEXT_LENGTH, D_IN)


# ── Single Self-Attention Block ──────────────────────────────────────────────
class SelfAttention(nn.Module):
    """
    Scaled dot-product self-attention.
    W_q, W_k, W_v are learned linear projections (no bias, following GPT convention).
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        Q = self.W_query(x)                            # (batch, seq, d_out)
        K = self.W_key(x)                              # (batch, seq, d_out)
        V = self.W_value(x)                            # (batch, seq, d_out)

        # Scaled dot-product attention scores
        scale        = K.shape[-1] ** 0.5
        attn_scores  = (Q @ K.transpose(-2, -1)) / scale   # (batch, seq, seq)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum over values
        context_vecs = attn_weights @ V                # (batch, seq, d_out)
        return context_vecs, attn_weights


# ── Demo / training verification ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n\n________ SELF-ATTENTION FORWARD PASS ________\n")

    dataloader = create_dataloader_v1(
        raw_text, batch_size=2, max_length=CONTEXT_LENGTH,
        stride=CONTEXT_LENGTH, shuffle=False
    )
    inputs, _ = next(iter(dataloader))          # inputs: (batch=2, seq=4)

    # Build input embeddings  (token + positional)
    token_embeddings = token_emb_layer(inputs)                          # (2, 4, 256)
    pos_embeddings   = pos_emb_layer(torch.arange(CONTEXT_LENGTH))     # (4, 256)
    x = token_embeddings + pos_embeddings                               # (2, 4, 256)

    print("Input embedding shape  :", x.shape)

    # Forward pass through self-attention
    attn = SelfAttention(D_IN, D_OUT)
    context_vecs, attn_weights = attn(x)

    print("Context vector shape   :", context_vecs.shape)
    print("Attention weights shape:", attn_weights.shape)

    print("\nAttention weight matrix (batch 0) — rows are query tokens, cols are key tokens:")
    print(attn_weights[0].detach())

    print("\nContext vector for token 0 (batch 0) — first 8 dims:")
    print(context_vecs[0][0][:8].detach())

    # ── Verify QKV weights are trainable via backprop ────────────────────────
    print("\n\n________ TRAINING QKV WEIGHTS ________\n")

    optimizer = torch.optim.Adam(
        list(token_emb_layer.parameters()) +
        list(pos_emb_layer.parameters()) +
        list(attn.parameters()),
        lr=1e-3
    )

    for step in range(3):
        optimizer.zero_grad()

        token_embeddings = token_emb_layer(inputs)
        pos_embeddings   = pos_emb_layer(torch.arange(CONTEXT_LENGTH))
        x = token_embeddings + pos_embeddings

        context_vecs, attn_weights = attn(x)

        # Dummy loss: push context vectors toward zero (just to verify gradient flow)
        loss = context_vecs.pow(2).mean()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1} | loss = {loss.item():.6f}")

    print("\nTrainable parameters:")
    for name, param in attn.named_parameters():
        print(f"  {name:12s}  shape={tuple(param.shape)}  grad_fn={param.grad is not None}")

    # ── Save QKV weights ──────────────────────────────────────────────────────
    os.makedirs("Attention Weights", exist_ok=True)
    torch.save(
        {
            "W_query": attn.W_query.state_dict(),
            "W_key":   attn.W_key.state_dict(),
            "W_value": attn.W_value.state_dict(),
        },
        os.path.join("Attention Weights", "self_attention_weights.pt"),
    )
    print("\nQKV weights saved → Attention Weights/self_attention_weights.pt")
