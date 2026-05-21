"""
models/image_encoder.py — ResNet-50 Visual Feature Extractor

KEY CONCEPTS
────────────
Transfer Learning:
    ResNet-50 was pretrained on ImageNet (1.2 M images, 1000 classes).
    Its early layers detect universal features — edges, textures, colour blobs.
    We reuse those and only fine-tune the deepest layers for our VQA task.

Spatial Feature Map vs. Global Pool:
    Standard ResNet ends with avgpool + fc → one class vector (no spatial info).
    We stop before avgpool, keeping the 7×7 feature grid (49 "visual tokens").
    Each token encodes what is in one 32×32 patch of the image.

Positional Embeddings:
    Without them the model cannot distinguish a circle at top-left from one
    at bottom-right — both produce the same token values.
    Learnable position embeddings add spatial awareness.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Converts a batch of RGB images into a sequence of visual tokens.

    Input  : (B, 3, 224, 224)
    Output : (B, 49, hidden_dim)   — 49 spatial tokens per image
    """

    def __init__(self, hidden_dim: int = 512, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 1. Load pretrained ResNet-50 ──────────────────────────────────
        # weights= argument preferred in torchvision ≥ 0.13; fall back to
        # the older pretrained= flag for backward compatibility.
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet  = models.resnet50(weights=weights)
        except AttributeError:
            resnet  = models.resnet50(pretrained=pretrained)

        # ResNet children (in order):
        #   [0] conv1  [1] bn1   [2] relu  [3] maxpool
        #   [4] layer1 [5] layer2 [6] layer3 [7] layer4
        #   [8] avgpool [9] fc          ← we drop these two
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ── 2. Selective freezing ─────────────────────────────────────────
        # Freeze everything first, then unfreeze only the deepest block.
        # Why? Early layers capture low-level features (edges, colours) that
        # generalise well. Deep layers are task-specific and worth adapting.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer4 (index 7) for light fine-tuning
            for param in self.backbone[7].parameters():
                param.requires_grad = True

        # ── 3. Projection into shared embedding space ─────────────────────
        # Maps ResNet's 2048-dim features → hidden_dim (default 512).
        # LayerNorm stabilises training; Dropout prevents over-fitting.
        self.projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # ── 4. Learnable positional embeddings ────────────────────────────
        # 49 positions (7×7 grid), each represented by a hidden_dim vector.
        # Registered as a buffer → saved with the model but not optimised.
        self.position_embedding = nn.Embedding(49, hidden_dim)
        self.register_buffer("positions", torch.arange(49))

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) — normalised RGB tensor

        Returns:
            features: (B, 49, hidden_dim) — sequence of visual tokens
        """
        B = images.size(0)

        # Extract spatial feature map: (B, 2048, 7, 7)
        feat_map = self.backbone(images)

        # Flatten the 7×7 grid into 49 tokens: (B, 49, 2048)
        feat_map = feat_map.permute(0, 2, 3, 1).reshape(B, 49, 2048)

        # Project to hidden dimension: (B, 49, hidden_dim)
        features = self.projection(feat_map)

        # Add positional embeddings (broadcast over batch)
        features = features + self.position_embedding(self.positions)

        return features

    # ──────────────────────────────────────────────────────────────────────
    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
