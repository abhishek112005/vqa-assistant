"""
models/clip_encoder.py — CLIP-based Image and Text Encoders

Why CLIP instead of ResNet + BERT?
────────────────────────────────────
ResNet was trained to classify 1000 ImageNet categories.
BERT was trained to understand text.
They live in completely SEPARATE spaces — fusion must bridge a large gap.

CLIP (Contrastive Language-Image Pretraining, OpenAI 2021) was trained
on 400 MILLION (image, caption) pairs from the internet using contrastive
loss: push matching image-text pairs close, push non-matching pairs apart.

Result: image features and text features already live in the SAME
512-dimensional space. Fusion is much easier — the gap is already bridged.

Architecture used: clip-vit-base-patch32
    Vision: ViT-B/32 → 49 patch tokens of 768-dim → project to 512-dim
    Text  : Transformer → token embeddings of 512-dim
    Both  → same joint embedding space

Because CLIP was trained on real internet images, its features generalise
to photos, diagrams, charts, and real-world scenes out of the box.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPImageEncoder(nn.Module):
    """
    Extracts spatial patch features from images using CLIP ViT-B/32.

    Input  : (B, 3, 224, 224)
    Output : (B, 49, hidden_dim)   — 49 patch tokens (7×7 grid)

    The CLIP vision transformer splits each 224×224 image into
    49 non-overlapping 32×32 patches. Each patch becomes one token.
    The internal hidden size is 768-dim; we project to hidden_dim (512).
    """

    def __init__(self, clip_model: CLIPModel, hidden_dim: int = 512):
        super().__init__()
        self.vision_model = clip_model.vision_model
        vision_hidden = clip_model.config.vision_config.hidden_size  # 768

        # Learnable projection: 768 → hidden_dim
        # Initialised with Xavier uniform for stable gradient flow.
        self.projection = nn.Sequential(
            nn.Linear(vision_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images : (B, 3, 224, 224) — preprocessed by CLIPProcessor

        Returns:
            features : (B, 49, hidden_dim)
        """
        # last_hidden_state: (B, 50, 768)  [0]=CLS, [1..49]=patches
        vision_out = self.vision_model(pixel_values=images)
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # (B, 49, 768)
        return self.projection(patch_tokens)                    # (B, 49, D)


class CLIPTextEncoder(nn.Module):
    """
    Encodes questions using CLIP's text transformer.

    Input  : tokenised question (input_ids, attention_mask)
    Output : (B, seq_len, hidden_dim)

    CLIP uses a BPE tokeniser (max 77 tokens, vocabulary 49K).
    The text transformer hidden size is 512-dim for ViT-B/32 —
    same as our hidden_dim, so the projection is almost an identity.
    """

    def __init__(self, clip_model: CLIPModel, hidden_dim: int = 512):
        super().__init__()
        self.text_model = clip_model.text_model
        text_hidden = clip_model.config.text_config.hidden_size  # 512

        self.projection = nn.Sequential(
            nn.Linear(text_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Load the matching tokeniser
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids      : (B, seq_len)
            attention_mask : (B, seq_len)

        Returns:
            features       : (B, seq_len, hidden_dim)
            attention_mask : unchanged (passed to fusion layer)
        """
        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_features = text_out.last_hidden_state    # (B, L, 512)
        return self.projection(token_features), attention_mask

    def tokenize(self, questions: list[str], max_length: int = 77,
                 device: torch.device = None) -> dict:
        """Tokenise questions using CLIP's BPE tokeniser."""
        enc = self.tokenizer(
            questions,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc


def load_clip_backbone(model_id: str = CLIP_MODEL_ID,
                       freeze: bool = True) -> CLIPModel:
    """
    Load the shared CLIP model once, freeze it, and return it.
    Both CLIPImageEncoder and CLIPTextEncoder accept this object
    to avoid loading the weights twice.
    """
    clip = CLIPModel.from_pretrained(model_id)
    if freeze:
        for param in clip.parameters():
            param.requires_grad = False
    return clip
