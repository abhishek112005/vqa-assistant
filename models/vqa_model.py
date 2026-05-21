"""
models/vqa_model.py — Complete Visual Question Answering Model

Overall data-flow
─────────────────
Image (B,3,224,224)  ──► ImageEncoder ──► (B, 49, 512)  ──┐
                                                            ├──► CrossAttentionFusion
Question tokens       ──► TextEncoder  ──► (B, L,  512)  ──┘
                                                            │
                                               (B, 49, 512) fused image
                                               (B, L,  512) fused text
                                                            │
                                                     AnswerHead
                                                            │
                                               (B, num_answers) logits
"""

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .text_encoder  import TextEncoder
from .fusion        import CrossAttentionFusion
from .clip_encoder  import CLIPImageEncoder, CLIPTextEncoder, load_clip_backbone


class AnswerHead(nn.Module):
    """
    Classifies the fused multimodal representation into one answer class.

    Strategy:
        • Pool image tokens  → mean over the 49 spatial positions
        • Take text [CLS]    → first token (index 0) summarises the question
        • Concatenate both   → (B, 2 × hidden_dim)
        • Two-layer MLP with GELU → (B, num_answers)

    Why concatenate instead of add?
        Addition forces image and text representations to occupy the same
        subspace, which is a strong constraint.  Concatenation lets the MLP
        learn any interaction between the two modalities.
    """

    def __init__(self, hidden_dim: int = 512, num_answers: int = 35,
                 dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_answers),
        )

    def forward(self, img_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features  : (B, 49, D) — fused image tokens
            text_features : (B,  L, D) — fused text tokens

        Returns:
            logits : (B, num_answers) — unnormalised class scores
        """
        # Mean-pool over the 49 visual tokens → (B, D)
        img_pooled  = img_features.mean(dim=1)

        # [CLS] token = position 0 → (B, D)
        text_cls    = text_features[:, 0, :]

        # Concatenate → (B, 2D) then classify
        combined = torch.cat([img_pooled, text_cls], dim=-1)
        return self.classifier(combined)


class VQAModel(nn.Module):
    """
    End-to-end Visual Question Answering model.

    Supports two backbones:
        backbone="resnet_bert" — ResNet-50 + BERT (original, trained on synthetic shapes)
        backbone="clip"        — CLIP ViT-B/32 (recommended, works on real images)

    Usage:
        model = VQAModel(num_answers=35, backbone="clip")
        logits, img_attn, text_attn = model(images, input_ids, attn_mask)
        pred   = logits.argmax(dim=-1)
    """

    def __init__(self, num_answers: int, hidden_dim: int = 512,
                 num_heads: int = 8, num_fusion_layers: int = 2,
                 dropout: float = 0.3, pretrained: bool = True,
                 backbone: str = "clip"):
        super().__init__()
        self.backbone = backbone

        if backbone == "clip":
            # Load CLIP once, share weights between both encoders.
            # Freeze CLIP completely — only fusion + head are trained.
            # This is the KEY improvement: CLIP already understands real images.
            print("  Using CLIP ViT-B/32 backbone (recommended for real images)")
            clip_model = load_clip_backbone(freeze=True)
            self.image_encoder = CLIPImageEncoder(clip_model, hidden_dim)
            self.text_encoder  = CLIPTextEncoder(clip_model, hidden_dim)
            self._clip_ref     = clip_model   # prevent garbage collection
        else:
            # Original ResNet-50 + BERT backbone (kept for educational comparison)
            print("  Using ResNet-50 + BERT backbone")
            self.image_encoder = ImageEncoder(
                hidden_dim=hidden_dim,
                pretrained=pretrained,
                freeze_backbone=True,
            )
            self.text_encoder = TextEncoder(
                hidden_dim=hidden_dim,
                pretrained=pretrained,
                freeze_layers=10,
            )
        self.fusion = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
        )
        self.answer_head = AnswerHead(
            hidden_dim=hidden_dim,
            num_answers=num_answers,
            dropout=dropout,
        )

        # Store for external use (e.g. checkpoint metadata)
        self.hidden_dim        = hidden_dim
        self.num_answers       = num_answers
        self.num_fusion_layers = num_fusion_layers

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor
                ) -> tuple[torch.Tensor, list, list]:
        """
        Args:
            images         : (B, 3, 224, 224) — normalised RGB images
            input_ids      : (B, seq_len)      — BERT token indices
            attention_mask : (B, seq_len)      — 1=real token, 0=padding

        Returns:
            logits          : (B, num_answers) — answer class scores
            img_attn_weights: list of per-layer (B, 49, seq_len) attention maps
            text_attn_weights: list of per-layer (B, seq_len, 49) attention maps
        """
        # ── Encode each modality independently ────────────────────────────
        img_features  = self.image_encoder(images)          # (B, 49, D)
        text_features, text_mask = self.text_encoder(       # (B, L, D)
            input_ids, attention_mask
        )

        # ── Fuse modalities via bidirectional cross-attention ──────────────
        img_fused, text_fused, img_attn, text_attn = self.fusion(
            img_features, text_features, text_mask
        )

        # ── Classify into an answer ────────────────────────────────────────
        logits = self.answer_head(img_fused, text_fused)

        return logits, img_attn, text_attn

    # ──────────────────────────────────────────────────────────────────────
    def predict(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[list[int], list[float]]:
        """
        Convenience method: returns (predicted_indices, confidence_scores).
        Applies softmax so confidences sum to 1.
        """
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(images, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return preds.tolist(), probs.max(dim=-1).values.tolist()

    # ──────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> dict:
        """Return parameter counts per sub-module."""
        def _count(module):
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters()
                           if p.requires_grad)
            return total, trainable

        modules = {
            "image_encoder": self.image_encoder,
            "text_encoder" : self.text_encoder,
            "fusion"       : self.fusion,
            "answer_head"  : self.answer_head,
        }
        counts = {}
        grand_total, grand_train = 0, 0
        for name, mod in modules.items():
            t, tr = _count(mod)
            counts[name] = {"total": t, "trainable": tr}
            grand_total += t
            grand_train += tr
        counts["TOTAL"] = {"total": grand_total, "trainable": grand_train}
        return counts

    # ──────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str, epoch: int, optimizer=None,
                        val_acc: float = 0.0):
        """Save model state + metadata to a .pth file."""
        checkpoint = {
            "epoch"            : epoch,
            "val_acc"          : val_acc,
            "model_state"      : self.state_dict(),
            "num_answers"      : self.num_answers,
            "hidden_dim"       : self.hidden_dim,
            "num_fusion_layers": self.num_fusion_layers,
            "backbone"         : self.backbone,
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved → {path}  (epoch {epoch}, val_acc {val_acc:.4f})")

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None):
        """Load a saved checkpoint and return (model, epoch, val_acc)."""
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_answers       = ckpt["num_answers"],
            hidden_dim        = ckpt["hidden_dim"],
            num_fusion_layers = ckpt["num_fusion_layers"],
            backbone          = ckpt.get("backbone", "resnet_bert"),
            pretrained        = True,   # CLIP needs pretrained weights to load encoders
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        print(f"  ✓ Checkpoint loaded ← {path}  "
              f"(backbone={ckpt.get('backbone','resnet_bert')}, "
              f"epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
        return model, ckpt["epoch"], ckpt["val_acc"]
