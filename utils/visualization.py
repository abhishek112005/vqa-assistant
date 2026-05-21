"""
utils/visualization.py — Attention & Grad-CAM Visualisation

Visualising WHAT the model looks at when answering a question is one
of the most educational features of an attention-based system.

Attention Visualisation:
    The cross-attention weights (B, Lq, Ls) tell us, for each image token
    (query), how much it attended to each text token (source), and vice versa.
    We reshape image attention back to the 7×7 spatial grid and overlay it
    on the original image as a heatmap.

Grad-CAM (Gradient-weighted Class Activation Mapping):
    Instead of attention weights, Grad-CAM uses GRADIENTS flowing back into
    the final convolutional feature map to highlight discriminative regions.
    Algorithm:
        1. Forward pass → get logits for target class.
        2. Backward pass → compute dL/dA (gradient w.r.t. conv feature map).
        3. Global average pool the gradients: α_k = mean(dL/dA_k)  per channel.
        4. Weighted sum: CAM = ReLU( Σ_k  α_k · A_k )
        5. Upsample to image size and overlay.
    Grad-CAM requires no architectural changes and works on any CNN.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Attention heatmap
# ─────────────────────────────────────────────────────────────

def visualize_attention(image_tensor: torch.Tensor,
                        attn_weights: torch.Tensor,
                        question_tokens: list[str] = None,
                        title: str = "Cross-Attention") -> plt.Figure:
    """
    Overlay the image-side cross-attention heatmap on the original image.

    Args:
        image_tensor : (3, 224, 224) normalised image tensor
        attn_weights : (49, seq_len) attention weights for one sample
                       rows = image tokens, cols = text tokens
        question_tokens: list of decoded token strings (optional)
        title        : figure title

    Returns a matplotlib Figure.
    """
    from .preprocessing import denormalize_image

    # De-normalise for display
    img_np = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # Sum attention over all text tokens → one scalar per image token (49,)
    attn = attn_weights.detach().cpu().float()  # (49, L)
    heat = attn.mean(dim=-1)                     # (49,) — mean over text dim
    heat = heat.reshape(7, 7)                    # back to spatial grid

    # Normalise to [0, 1]
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    heat_np = heat.numpy()

    # Upsample to image resolution (224×224) with bilinear interpolation
    heat_tensor = torch.tensor(heat_np).unsqueeze(0).unsqueeze(0)  # (1,1,7,7)
    heat_up = F.interpolate(heat_tensor, size=(224, 224),
                             mode="bilinear", align_corners=False)
    heat_up = heat_up.squeeze().numpy()  # (224, 224)

    # Compose figure
    ncols = 3 if question_tokens is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(heat_up, alpha=0.55, cmap="jet")
    axes[1].set_title("Attention Heatmap", fontsize=11)
    axes[1].axis("off")

    if question_tokens is not None and len(question_tokens) > 0:
        # Bar chart: text-token attention averaged over image tokens
        token_attn = attn.mean(dim=0).numpy()  # (L,)
        # Trim to actual non-padding tokens
        tokens = question_tokens[:len(token_attn)]
        axes[2].barh(range(len(tokens)), token_attn[:len(tokens)],
                     color="steelblue")
        axes[2].set_yticks(range(len(tokens)))
        axes[2].set_yticklabels(tokens, fontsize=8)
        axes[2].invert_yaxis()
        axes[2].set_title("Token Attention", fontsize=11)
        axes[2].set_xlabel("Mean attention weight")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for the ResNet-50 backbone inside ImageEncoder.

    Hooks into the last convolutional block (layer4) to capture:
        • forward activations  A  of shape (B, 2048, 7, 7)
        • backward gradients   G  of shape (B, 2048, 7, 7)

    Usage:
        gcam = GradCAM(model)
        cam  = gcam.generate(images, input_ids, attn_mask, target_class_idx)
        fig  = gcam.overlay(original_image, cam)
        gcam.remove_hooks()
    """

    def __init__(self, vqa_model):
        self.model      = vqa_model
        self._activations = None
        self._gradients   = None

        # Hook onto the last conv block in ResNet (layer4 = backbone[7])
        target_layer = vqa_model.image_encoder.backbone[7]
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, images: torch.Tensor,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Returns:
            cam_up : (224, 224) numpy array in [0, 1]
        """
        self.model.eval()
        images = images.requires_grad_(False)

        # Forward pass
        logits, _, _ = self.model(images, input_ids, attention_mask)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        # Backward pass on the target class score
        self.model.zero_grad()
        logits[0, target_class].backward()

        # α_k = global average of gradients over spatial dims
        grads = self._gradients  # (B, 2048, 7, 7)
        alpha = grads.mean(dim=(2, 3), keepdim=True)  # (B, 2048, 1, 1)

        # Weighted combination of activations
        activations = self._activations  # (B, 2048, 7, 7)
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # (B, 1, 7, 7)
        cam = F.relu(cam)                  # only keep positive contributions

        # Upsample to 224×224
        cam_up = F.interpolate(cam, size=(224, 224),
                               mode="bilinear", align_corners=False)
        cam_up = cam_up.squeeze().cpu().numpy()  # (224, 224)

        # Normalise
        cam_up = cam_up - cam_up.min()
        cam_up = cam_up / (cam_up.max() + 1e-8)
        return cam_up

    @staticmethod
    def overlay(pil_image: Image.Image, cam: np.ndarray,
                alpha: float = 0.5, title: str = "Grad-CAM") -> plt.Figure:
        """
        Overlay Grad-CAM heatmap on the original PIL image.

        Returns a matplotlib Figure.
        """
        img_np = np.array(pil_image.resize((224, 224)).convert("RGB")) / 255.0

        colormap = cm.get_cmap("jet")
        heatmap  = colormap(cam)[:, :, :3]   # (224, 224, 3) — drop alpha
        blended  = alpha * heatmap + (1 - alpha) * img_np
        blended  = np.clip(blended, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np);   axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM Map"); axes[1].axis("off")
        axes[2].imshow(blended);  axes[2].set_title("Overlay");  axes[2].axis("off")

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def grad_cam_heatmap(vqa_model, image_tensor: torch.Tensor,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     pil_image: Image.Image,
                     target_class: int = None) -> plt.Figure:
    """
    Convenience wrapper: compute and overlay Grad-CAM in one call.
    """
    gcam = GradCAM(vqa_model)
    try:
        cam = gcam.generate(image_tensor, input_ids, attention_mask, target_class)
        fig = GradCAM.overlay(pil_image, cam, title="Grad-CAM Visualisation")
    finally:
        gcam.remove_hooks()
    return fig
