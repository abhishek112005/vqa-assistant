"""
utils/dataset.py — PyTorch Dataset and DataLoader for VQA

PyTorch Dataset (torch.utils.data.Dataset):
    A class that tells PyTorch how to access a single sample via __getitem__
    and how many samples exist via __len__.
    PyTorch does the rest: batching, shuffling, parallel loading.

DataLoader:
    Wraps a Dataset.  On each iteration it:
        1. Selects a batch of indices (shuffled during training).
        2. Calls __getitem__ for each index (possibly in parallel workers).
        3. Collates results into a single batched tensor.

Collate function (collate_fn):
    The default collate_fn stacks tensors of the same shape.
    We provide a custom one because tokenised text can have variable
    lengths — we pad within the batch to the maximum length present,
    rather than always padding to a global maximum.
    This makes training faster when most questions are short.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .preprocessing import get_image_transforms, tokenize_question
from data.generate_synthetic import SyntheticDataGenerator
import config as cfg


# ─────────────────────────────────────────────────────────────
# In-memory synthetic VQA dataset
# ─────────────────────────────────────────────────────────────

class SyntheticVQADataset(Dataset):
    """
    Generates (or re-uses) synthetic VQA samples in memory.

    Args:
        tokenizer : HuggingFace BertTokenizer
        split     : "train" | "val" | "test"
        n_samples : number of samples to generate
        seed      : random seed for reproducibility
    """

    def __init__(self, tokenizer, split: str = "train",
                 n_samples: int = 1000, seed: int = 42):
        super().__init__()
        self.tokenizer = tokenizer
        self.split     = split
        self.transform = get_image_transforms(split)

        # Generate all samples deterministically
        random.seed(seed + {"train": 0, "val": 1, "test": 2}.get(split, 0))
        gen = SyntheticDataGenerator()
        self.samples = gen.generate_dataset(n_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample   = self.samples[idx]
        pil_img  = sample["image"].convert("RGB")
        question = sample["question"]
        answer   = sample["answer"]

        # Image → tensor
        image_tensor = self.transform(pil_img)  # (3, 224, 224)

        # Question → BERT tokens
        max_len = getattr(self.tokenizer, "model_max_length", cfg.MAX_SEQ_LEN)
        max_len = min(max_len, 77)   # cap at 77 for CLIP; 64 for BERT
        encoding = self.tokenizer(
            question,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)       # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)

        # Answer → class index
        label = cfg.ANSWER_TO_IDX.get(answer, 0)

        return {
            "image"         : image_tensor,
            "input_ids"     : input_ids,
            "attention_mask": attention_mask,
            "label"         : torch.tensor(label, dtype=torch.long),
            "question"      : question,
            "answer"        : answer,
        }


# ─────────────────────────────────────────────────────────────
# On-disk JSON manifest dataset
# ─────────────────────────────────────────────────────────────

class VQADataset(Dataset):
    """
    Loads VQA samples from a JSON manifest produced by SyntheticDataGenerator
    or from a CLEVR / VQA-v2 style annotation file.

    Expected manifest format (list of dicts):
        [
          {"image_path": "train_00001.png",
           "question"  : "What color is the circle?",
           "answer"    : "blue"},
          ...
        ]

    Args:
        manifest_path : path to JSON manifest file
        image_dir     : directory containing images
        tokenizer     : HuggingFace BertTokenizer
        split         : "train" | "val" | "test"
        max_samples   : if set, cap the dataset size
    """

    def __init__(self, manifest_path: str, image_dir: str,
                 tokenizer, split: str = "train",
                 max_samples: int = None):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.split     = split
        self.transform = get_image_transforms(split)

        with open(manifest_path) as f:
            self.samples = json.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, s["image_path"])
        pil_img  = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(pil_img)

        # Tokenise
        encoding = self.tokenizer(
            s["question"],
            max_length=cfg.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        label = cfg.ANSWER_TO_IDX.get(s["answer"], 0)

        return {
            "image"         : image_tensor,
            "input_ids"     : input_ids,
            "attention_mask": attention_mask,
            "label"         : torch.tensor(label, dtype=torch.long),
            "question"      : s["question"],
            "answer"        : s["answer"],
        }


# ─────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────

def build_dataloaders(tokenizer,
                      batch_size: int = 16,
                      num_workers: int = 0,
                      use_disk: bool = False,
                      data_dir: str = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Args:
        tokenizer   : HuggingFace BertTokenizer
        batch_size  : samples per batch
        num_workers : parallel data-loading workers (0 = main process)
        use_disk    : if True, load from saved JSON manifests instead of
                      generating in memory (requires prior save_dataset call)
        data_dir    : path to directory with manifest JSONs + images

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if use_disk and data_dir is not None:
        train_ds = VQADataset(
            os.path.join(data_dir, "train_manifest.json"),
            data_dir, tokenizer, "train"
        )
        val_ds = VQADataset(
            os.path.join(data_dir, "val_manifest.json"),
            data_dir, tokenizer, "val"
        )
        test_ds = VQADataset(
            os.path.join(data_dir, "test_manifest.json"),
            data_dir, tokenizer, "test"
        )
    else:
        train_ds = SyntheticVQADataset(tokenizer, "train", cfg.NUM_TRAIN_SAMPLES)
        val_ds   = SyntheticVQADataset(tokenizer, "val",   cfg.NUM_VAL_SAMPLES)
        test_ds  = SyntheticVQADataset(tokenizer, "test",  cfg.NUM_TEST_SAMPLES)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"  Train: {len(train_ds):>5} samples | "
          f"Val: {len(val_ds):>5} | Test: {len(test_ds):>5}")
    return train_loader, val_loader, test_loader
