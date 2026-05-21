---
title: Visual Question Answering Assistant
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# 🧠 Educational Visual Question Answering Assistant

> A lightweight multimodal AI system that accepts an image and a natural-language
> question, then generates a short answer — built for learning, not competition.

---

## 🎯 Project Objective

This project demonstrates core ideas in modern AI:

| Concept | Where it appears |
|---------|-----------------|
| **Computer Vision** | ResNet-50 feature extraction |
| **NLP** | BERT tokenisation + contextual embeddings |
| **Transformers** | Cross-attention fusion layers |
| **Attention** | Image ↔ text bidirectional attention |
| **Multimodal AI** | Joint image–text representation |
| **Transfer Learning** | Pretrained ResNet + BERT, frozen early layers |

The goal is **clean architecture** and **educational clarity** — not beating GPT-4V.

---

## 🏗️ Architecture

```
Image (224×224)          Question (text)
      │                        │
  ResNet-50               BERT Encoder
  Backbone                (bert-base)
  (frozen)                (frozen)
      │                        │
(B, 49, 512)            (B, L, 512)
      │                        │
      └──────┬─────────────────┘
             │
    Cross-Attention Fusion (×2 layers)
    ┌────────────────────────────────┐
    │  Image ←cross-attn← Text      │  image patches attend to words
    │  Text  ←cross-attn← Image     │  words attend to image regions
    └────────────────────────────────┘
             │
    Mean-pool image + [CLS] text → concat
             │
        MLP Classifier
             │
       Answer Logits (35 classes)
```

### Components

**1. Image Encoder** (`models/image_encoder.py`)
- ResNet-50 pretrained on ImageNet (1.2M images)
- Removes final avgpool+fc → keeps 7×7 spatial feature maps
- Flattens to 49 visual tokens of 512-dim each
- Adds learnable 2D positional embeddings
- layer4 is fine-tunable; earlier layers frozen

**2. Text Encoder** (`models/text_encoder.py`)
- `bert-base-uncased`: 12 Transformer layers, 768-dim hidden
- Produces contextual token embeddings (one per WordPiece token)
- Projected from 768 → 512-dim to match image features
- First 10 layers frozen; last 2 fine-tunable

**3. Cross-Attention Fusion** (`models/fusion.py`)
- Bidirectional: image→text cross-attention + text→image cross-attention
- Each direction has self-attention + FFN + LayerNorm
- 8 heads, GELU activation, 2 stacked layers
- Returns attention weights for visualisation

**4. Answer Head** (`models/vqa_model.py → AnswerHead`)
- Mean-pool 49 image tokens → 512-dim
- Take [CLS] text token       → 512-dim
- Concatenate → 1024-dim → MLP (512 → 256 → 35)
- Softmax over 35 answer classes

---

## 📁 Project Structure

```
Visual Question Answering Assistant/
├── config.py               # All hyperparameters and paths
├── train.py                # Training script
├── infer.py                # Single-image inference
├── evaluate.py             # Full test-set evaluation + metrics
├── app.py                  # Streamlit web application
├── requirements.txt
├── README.md
│
├── models/
│   ├── image_encoder.py    # ResNet-50 visual feature extractor
│   ├── text_encoder.py     # BERT question encoder
│   ├── fusion.py           # Cross-attention fusion module
│   └── vqa_model.py        # Complete VQA model + checkpoint I/O
│
├── utils/
│   ├── dataset.py          # PyTorch Dataset & DataLoader builders
│   ├── preprocessing.py    # Image transforms + BERT tokenisation
│   ├── metrics.py          # Accuracy, confusion matrix, VQA score
│   └── visualization.py    # Attention heatmaps + Grad-CAM
│
├── data/
│   ├── generate_synthetic.py   # Synthetic coloured-shape dataset
│   └── sample_data/            # Saved images (after generate)
│
├── checkpoints/            # Saved model weights (.pth)
├── frontend/               # Extra CSS / static assets
└── notebooks/              # Jupyter exploration notebooks
```

---

## ⚙️ Setup

### Prerequisites
- Python ≥ 3.10
- pip
- (Optional) CUDA-enabled GPU

### Installation

```bash
# 1. Clone / download the project
cd "Visual Question Answering Assistant"

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU users**: visit [pytorch.org](https://pytorch.org) to install the
> CUDA-enabled PyTorch wheel first, then re-run `pip install -r requirements.txt`.

---

## 🚀 Quick Start (No Training Required)

The Streamlit app ships with a **Quick Demo** mode powered by
`dandelin/vilt-b32-finetuned-vqa` (downloaded automatically from HuggingFace):

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your browser.
2. Select **🚀 Quick Demo** in the sidebar.
3. Upload any image or tick **Generate a synthetic example**.
4. Type a question and click **Predict Answer**.

---

## 🏋️ Training the Custom Model

```bash
# Generate the synthetic dataset and train
python train.py

# With custom settings
python train.py --epochs 20 --batch 32
```

Training progress is printed to the console and saved to
`checkpoints/training_curves.png`.

Default settings (configurable in `config.py`):
| Setting | Value |
|---------|-------|
| Train samples | 5,000 |
| Val samples | 1,000 |
| Batch size | 16 |
| Epochs | 15 |
| Backbone LR | 1e-5 |
| Fusion LR | 1e-4 |
| Scheduler | OneCycleLR (cosine) |

---

## 🔍 Inference

```bash
# Run on a specific image
python infer.py --image path/to/image.png --question "What color is the circle?"

# Run demo on 5 auto-generated examples
python infer.py --demo --n 5
```

---

## 📊 Evaluation

```bash
# Evaluate on the test set
python evaluate.py

# With zero-shot evaluation and per-class metrics
python evaluate.py --zero-shot --verbose
```

Outputs:
- Console table of sample predictions
- `checkpoints/confusion_matrix.png`
- `checkpoints/predictions.csv`
- `checkpoints/zero_shot_results.csv` (if `--zero-shot`)

---

## 🌐 Streamlit App Features

| Feature | Description |
|---------|-------------|
| Image upload | PNG / JPG / BMP / WebP |
| Synthetic generator | Creates a test image + question on the fly |
| Quick Demo mode | ViLT (no training needed) |
| Custom model mode | Your trained ResNet+BERT model |
| Top-5 answers | Confidence bars for each candidate |
| Attention heatmap | Shows which image regions were attended to |
| Grad-CAM | Gradient-based discriminative localisation |
| Architecture tab | Interactive explanation of the model |

---

## 🗃️ Dataset

The project ships with a **synthetic dataset generator** (`data/generate_synthetic.py`)
that creates coloured geometric shapes paired with questions:

| Type | Example Q | Example A |
|------|-----------|-----------|
| Colour | "What color is the circle?" | "blue" |
| Shape | "What shape is the red object?" | "square" |
| Count | "How many shapes are there?" | "3" |
| Yes/No | "Is there a green triangle?" | "yes" |

Answer vocabulary (35 classes): `yes`, `no`, 10 colours, 4 shapes,
counts 0–5, 4 circuit components, size/direction words.

To use a real dataset (CLEVR, VQA-v2), save a JSON manifest in the format
`[{"image_path": "...", "question": "...", "answer": "..."}, ...]`
and point `VQADataset` at it in `utils/dataset.py`.

---

## 📸 Screenshots

| Upload & Predict | Attention Heatmap | Architecture |
|:----------------:|:-----------------:|:------------:|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 🗺️ Future Improvements

- [ ] Train on VQA-v2 or CLEVR (larger, more diverse data)
- [ ] Add visual grounding (bounding-box output)
- [ ] Replace ResNet-50 with Vision Transformer (ViT-B/16)
- [ ] Use BLIP-2 Q-Former fusion instead of custom cross-attention
- [ ] Export to ONNX for browser inference
- [ ] Add chatbot-style multi-turn conversation history
- [ ] Fine-tune with RLHF for better answer phrasing

---

## 🎤 Interview Explanation

> *"How would you explain this project in 2 minutes?"*

**Problem**: Given an image and a question, predict the answer.

**Approach**:
1. Extract **visual features** using ResNet-50 (pretrained, mostly frozen).
   Instead of a single global vector, I keep the 7×7 spatial grid — 49 tokens
   that tell us *where* things are, not just *what* is there.

2. Extract **text features** using BERT (pretrained, mostly frozen).
   BERT gives contextual embeddings — "bank" means something different next to
   "river" vs. "money".

3. **Fuse** both modalities with bidirectional cross-attention:
   image tokens attend to text tokens (which words describe this patch?),
   text tokens attend to image tokens (which region does this word refer to?).

4. **Classify** by concatenating the mean-pooled image representation and the
   [CLS] text summary, then passing through a small MLP.

**Key design choices**:
- Separate learning rates: pretrained layers get 1e-5, new layers get 1e-4.
  Prevents catastrophic forgetting while still adapting to the task.
- Label smoothing (ε=0.1): prevents the model from becoming overconfident.
- OneCycleLR with cosine annealing: stable training with warm-up.

---

## 📄 Licence

MIT — free to use for educational and personal projects.

---

*Built with PyTorch · HuggingFace Transformers · Streamlit*
