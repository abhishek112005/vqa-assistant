"""
models/fusion.py — Cross-Attention Multimodal Fusion

KEY CONCEPTS
────────────
Attention (Scaled Dot-Product):
    Given Query (Q), Key (K), Value (V) matrices:
        Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
    The weights tell the model "how much to look at" each value.
    √d_k scaling prevents vanishing gradients when d_k is large.

Multi-Head Attention:
    Run attention h times in parallel with different learned projections.
    Each "head" can focus on a different relationship pattern.
    Outputs are concatenated and projected back to hidden_dim.

Cross-Attention (the key idea in multimodal fusion):
    In self-attention Q=K=V (a sequence attends to itself).
    In cross-attention Q comes from one modality (e.g. image tokens)
    while K and V come from another (e.g. text tokens).
    → Image tokens learn WHICH words are relevant for each image patch.
    → Text tokens learn WHICH image regions support each word.

Residual Connection + LayerNorm:
    output = LayerNorm(x + Sublayer(x))
    Residuals let gradients flow directly to early layers (no vanishing).
    LayerNorm normalises across the feature dimension for stable training.

Feed-Forward Network (FFN):
    A two-layer MLP applied independently to each token:
        FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂
    The hidden size is 4× the model size (standard Transformer ratio).
    GELU is smoother than ReLU; used in BERT and GPT-family models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise feed-forward sub-layer used inside each fusion block."""

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),                              # smooth non-linearity
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    One cross-attention fusion block.

    Performs:
        1. Cross-attention:  query modality attends to source modality
        2. Residual + LayerNorm
        3. Self-attention:   refined query tokens attend to each other
        4. Residual + LayerNorm
        5. Feed-forward network
        6. Residual + LayerNorm
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # Cross-attention: query_tokens ← source_tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # (B, seq, dim) convention
        )

        # Self-attention on the query sequence after cross-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

        # Layer normalisation — one per sub-layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, source: torch.Tensor,
                source_key_padding_mask: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query  : (B, Lq, D) — the modality we are updating
            source : (B, Ls, D) — the modality we are attending to
            source_key_padding_mask: (B, Ls) — True for positions to ignore

        Returns:
            updated query tokens and cross-attention weights (B, Lq, Ls)
        """
        # ── Cross-attention ───────────────────────────────────────────────
        # Q = query, K = V = source
        # attn_weights shape: (B, Lq, Ls)
        cross_out, attn_weights = self.cross_attn(
            query=query,
            key=source,
            value=source,
            key_padding_mask=source_key_padding_mask,
        )
        query = self.norm1(query + self.dropout(cross_out))

        # ── Self-attention ────────────────────────────────────────────────
        self_out, _ = self.self_attn(query=query, key=query, value=query)
        query = self.norm2(query + self.dropout(self_out))

        # ── Feed-forward ──────────────────────────────────────────────────
        query = self.norm3(query + self.ffn(query))

        return query, attn_weights


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion for image + text features.

    Architecture (repeated num_layers times):
        ┌──────────────────────────────────────────────┐
        │  Image tokens  ←cross-attn← Text tokens      │  image sees text
        │  Text tokens   ←cross-attn← Image tokens     │  text sees image
        └──────────────────────────────────────────────┘

    After all layers, image and text tokens are aligned in a shared
    multimodal embedding space, enriched by information from each other.
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers

        # Two stacks of cross-attention blocks — one per direction
        self.img_blocks  = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout)
             for _ in range(num_layers)]
        )
        self.text_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, img_features: torch.Tensor,
                text_features: torch.Tensor,
                text_mask: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor, list, list]:
        """
        Args:
            img_features  : (B, 49, D)       — visual tokens from ImageEncoder
            text_features : (B, seq_len, D)   — text tokens from TextEncoder
            text_mask     : (B, seq_len)       — attention_mask (1=real, 0=pad)

        Returns:
            img_features        : (B, 49, D)      — image-grounded by text
            text_features       : (B, seq_len, D) — text-grounded by image
            img_attn_weights    : list of (B, 49, seq_len)   per layer
            text_attn_weights   : list of (B, seq_len, 49)   per layer
        """
        # PyTorch's MultiheadAttention uses True to MASK (ignore) positions.
        # Our attention_mask uses 1=keep, 0=pad → invert.
        pad_mask = None
        if text_mask is not None:
            pad_mask = (text_mask == 0)   # (B, seq_len), True = padding

        img_attn_list  = []
        text_attn_list = []

        for i in range(self.num_layers):
            # Image tokens attend to text tokens
            img_features, img_w = self.img_blocks[i](
                query=img_features,
                source=text_features,
                source_key_padding_mask=pad_mask,
            )

            # Text tokens attend to (updated) image tokens
            text_features, text_w = self.text_blocks[i](
                query=text_features,
                source=img_features,
            )

            img_attn_list.append(img_w)
            text_attn_list.append(text_w)

        return img_features, text_features, img_attn_list, text_attn_list
