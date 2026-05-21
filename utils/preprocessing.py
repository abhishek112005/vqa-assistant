"""
utils/preprocessing.py — Image and Text Preprocessing Utilities

Why preprocessing matters
──────────────────────────
Neural networks expect inputs within a specific numerical range and format.
Raw images (0–255 pixels) and raw text (strings) must be converted to
normalised tensors before the model can process them.

Image preprocessing (standard ImageNet pipeline):
    1. Resize  → 224×224 (ResNet expected size)
    2. Convert → RGB tensor [0.0, 1.0]
    3. Normalise → subtract ImageNet mean, divide by std
       mean = [0.485, 0.456, 0.406]   (measured on ImageNet training set)
       std  = [0.229, 0.224, 0.225]
    These exact values must be used since the ResNet weights were trained
    with them — using wrong stats degrades feature quality significantly.

Text preprocessing (BERT pipeline):
    1. Tokenise with WordPiece → list of integer token IDs
    2. Add special tokens: [CLS] at start, [SEP] at end
    3. Pad or truncate to a fixed length
    4. Generate attention mask (1 = real token, 0 = padding)

Data augmentation (training only):
    Geometric and colour jitter transforms increase effective dataset size
    and improve generalisation by showing the model slightly varied inputs.
"""

import torch
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_image_transforms(split: str = "train",
                          image_size: int = 224) -> transforms.Compose:
    """
    Returns the torchvision transform pipeline for a given dataset split.

    Args:
        split      : "train" | "val" | "test"
        image_size : target spatial resolution (pixels)

    Training transforms add randomness for data augmentation.
    Val/Test transforms are deterministic for reproducible evaluation.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            # Random crop reintroduces some spatial variation
            transforms.RandomCrop(image_size),
            # Horizontal flip: shapes at left ↔ right are equally valid
            transforms.RandomHorizontalFlip(p=0.5),
            # Colour jitter: small brightness/contrast variation
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # For val/test: just resize + centre crop + normalise (no randomness)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def load_image(path_or_pil, split: str = "val") -> torch.Tensor:
    """
    Load an image from a file path or PIL object and apply transforms.

    Returns a (1, 3, H, W) tensor ready for the model (batch size = 1).
    """
    if isinstance(path_or_pil, (str, bytes)):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")

    transform = get_image_transforms(split)
    tensor = transform(img)          # (3, H, W)
    return tensor.unsqueeze(0)       # (1, 3, H, W)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation purposes.
    Input:  (C, H, W) normalised tensor
    Output: (C, H, W) tensor in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────

def tokenize_question(tokenizer, question: str | list[str],
                      max_length: int = 64,
                      device: torch.device = None) -> dict:
    """
    Tokenise one or more questions using the BERT tokeniser.

    BERT's WordPiece tokeniser:
        • Splits "playing" → ["play", "##ing"]   (##  = continuation piece)
        • Handles out-of-vocabulary words via sub-word decomposition
        • Always adds [CLS] (token 101) at position 0
        • Always adds [SEP] (token 102) at the end

    Args:
        tokenizer : a HuggingFace BertTokenizer instance
        question  : a single string or list of strings
        max_length: sequences are padded/truncated to this length
        device    : if given, tensors are moved to this device

    Returns:
        dict with 'input_ids' and 'attention_mask' tensors
    """
    if isinstance(question, str):
        question = [question]

    encoding = tokenizer(
        question,
        max_length=max_length,
        padding="max_length",     # pad short sequences with [PAD] tokens
        truncation=True,           # truncate long sequences
        return_tensors="pt",       # return PyTorch tensors
    )
    if device is not None:
        encoding = {k: v.to(device) for k, v in encoding.items()}
    return encoding


def decode_tokens(tokenizer, input_ids: torch.Tensor) -> list[str]:
    """
    Convert token index tensors back to human-readable strings.
    Useful for debugging the tokenisation output.
    """
    return [tokenizer.decode(ids, skip_special_tokens=True)
            for ids in input_ids]
