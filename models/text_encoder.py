"""
models/text_encoder.py — BERT-based Question Encoder

KEY CONCEPTS
────────────
Tokenisation (BPE / WordPiece):
    BERT splits words into sub-word pieces ("playing" → "play" + "##ing").
    This handles rare/unknown words gracefully and keeps the vocabulary small.

Contextual Embeddings:
    Unlike word2vec (one static vector per word), BERT produces embeddings
    that change based on surrounding context.
    "bank" means something different in "river bank" vs. "bank account".

[CLS] Token:
    BERT prepends a special [CLS] token to every input.
    After passing through all attention layers, the [CLS] hidden state
    aggregates the meaning of the whole sentence — used as the text summary.

Attention Mask:
    Padding tokens (added so all sequences in a batch have equal length)
    must be ignored. The mask (1 = real token, 0 = padding) achieves this.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """
    Encodes a natural-language question into a sequence of contextual vectors.

    Input  : tokenised question (input_ids, attention_mask)
    Output : (B, seq_len, hidden_dim)  — one vector per token
    """

    def __init__(self, hidden_dim: int = 512, pretrained: bool = True,
                 bert_model: str = "bert-base-uncased", freeze_layers: int = 10):
        """
        Args:
            hidden_dim    : output embedding dimension (must match ImageEncoder)
            pretrained    : load pretrained BERT weights
            bert_model    : HuggingFace model identifier
            freeze_layers : how many of BERT's 12 encoder layers to freeze
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 1. Tokeniser ──────────────────────────────────────────────────
        # The tokeniser must match the model (same vocabulary, same sub-word
        # splitting rules). Always load from the same model identifier.
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        # ── 2. BERT encoder ───────────────────────────────────────────────
        self.bert = (BertModel.from_pretrained(bert_model) if pretrained
                     else BertModel(BertModel.config_class()))

        # ── 3. Selective freezing ─────────────────────────────────────────
        # BERT has 12 Transformer encoder layers (layer[0] … layer[11]).
        # We freeze the first `freeze_layers` and fine-tune the rest.
        # Freezing saves GPU memory and prevents catastrophic forgetting of
        # the pretrained language representations.
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last (12 - freeze_layers) encoder blocks
        num_layers = len(self.bert.encoder.layer)
        for layer_idx in range(freeze_layers, num_layers):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = True

        # Also unfreeze the pooler so [CLS] gets trained
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # ── 4. Projection ─────────────────────────────────────────────────
        # BERT outputs 768-dim; we project to hidden_dim to match the
        # image encoder's output dimension (shared "multimodal space").
        self.projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids      : (B, seq_len) — token indices from the tokeniser
            attention_mask : (B, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            features       : (B, seq_len, hidden_dim) — contextual embeddings
            attention_mask : passed through unchanged (fusion layer needs it)
        """
        # Run BERT — returns BaseModelOutputWithPoolingAndCrossAttentions
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state: (B, seq_len, 768) — one vector per token
        token_embeddings = outputs.last_hidden_state

        # Project all tokens to hidden_dim
        features = self.projection(token_embeddings)  # (B, seq_len, hidden_dim)

        return features, attention_mask

    # ──────────────────────────────────────────────────────────────────────
    def tokenize(self, questions: list[str], max_length: int = 64,
                 device: torch.device = None) -> dict:
        """
        Convenience wrapper: tokenise a list of question strings.

        Returns a dict with 'input_ids' and 'attention_mask' tensors,
        ready to pass directly to forward().
        """
        encoding = self.tokenizer(
            questions,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        return encoding

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
