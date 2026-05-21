"""
config.py — Central configuration for the Educational VQA Assistant.

All hyperparameters, paths, and answer vocabulary live here so every
module stays in sync without hard-coded magic numbers.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────

IMAGE_SIZE       = 224    # ResNet expected input (pixels)
IMAGE_FEATURES   = 2048   # ResNet-50 last-conv output channels
NUM_VISUAL_TOKENS = 49    # 7×7 spatial grid flattened → 49 tokens

BERT_MODEL  = "bert-base-uncased"   # Pretrained BERT variant
BERT_HIDDEN = 768                   # BERT hidden dimension
MAX_SEQ_LEN = 64                    # Max question length (BPE tokens)

HIDDEN_DIM       = 512   # Shared embedding space for image + text
NUM_HEADS        = 8     # Attention heads in the fusion Transformer
NUM_FUSION_LAYERS = 2    # Cross-attention layers
DROPOUT          = 0.3   # Dropout rate for regularisation

# ─────────────────────────────────────────────────────────────
# ANSWER VOCABULARY
# VQA treats answer generation as classification over a fixed set.
# Each unique answer becomes a class the model learns to predict.
# ─────────────────────────────────────────────────────────────

ANSWER_VOCAB = [
    # ── yes / no ──────────────────────────────────────────────
    "yes", "no",
    # ── colours ───────────────────────────────────────────────
    "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "cyan", "white", "black", "gray", "brown",
    # ── shapes ────────────────────────────────────────────────
    "circle", "square", "triangle", "rectangle",
    # ── counts ────────────────────────────────────────────────
    "0", "1", "2", "3", "4", "5",
    # ── educational / circuit components ──────────────────────
    "capacitor", "resistor", "battery", "wire",
    # ── spatial / size ────────────────────────────────────────
    "large", "small", "left", "right",
    # ── chart types ───────────────────────────────────────────
    "bar chart", "pie chart", "graph",
]

NUM_ANSWERS    = len(ANSWER_VOCAB)
ANSWER_TO_IDX  = {a: i for i, a in enumerate(ANSWER_VOCAB)}
IDX_TO_ANSWER  = {i: a for i, a in enumerate(ANSWER_VOCAB)}

# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

BATCH_SIZE    = 16
NUM_EPOCHS    = 15
WARMUP_RATIO  = 0.1       # 10 % of steps used for LR warm-up
GRAD_CLIP     = 1.0       # Clip gradients to prevent explosion

# Separate learning rates: pretrained layers get a smaller LR
LR_BACKBONE   = 1e-5      # ResNet layer4 + BERT last 2 layers
LR_FUSION     = 1e-4      # New fusion + classification layers
WEIGHT_DECAY  = 1e-4

# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATASET SIZES
# ─────────────────────────────────────────────────────────────

NUM_TRAIN_SAMPLES = 5000
NUM_VAL_SAMPLES   = 1000
NUM_TEST_SAMPLES  = 500

# ─────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample_data")

os.makedirs(CHECKPOINT_DIR,  exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

CHECKPOINT_PATH      = os.path.join(CHECKPOINT_DIR, "vqa_best.pth")
CHECKPOINT_LAST_PATH = os.path.join(CHECKPOINT_DIR, "vqa_last.pth")

# ─────────────────────────────────────────────────────────────
# DEVICE — GPU when available, CPU otherwise
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
