"""
app.py — Streamlit Frontend for the Educational VQA Assistant

Two operating modes
────────────────────
1. Custom Model   (ResNet-50 + BERT fusion)
   Requires a trained checkpoint in checkpoints/vqa_best.pth.
   Run  python train.py  first.

2. Quick Demo     (ViLT — Vision-Language Transformer from HuggingFace)
   dandelin/vilt-b32-finetuned-vqa — a lightweight pretrained VQA model.
   Works immediately without any local training.
   Uses the HuggingFace pipeline API.

Run:
    streamlit run app.py
"""

import io
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image
import torch

# ─────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VQA Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(17,153,142,0.3);
        margin: 1rem 0;
    }
    /* Info card */
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    /* Confidence bar label */
    .conf-label {
        font-size: 0.85rem;
        color: #495057;
        margin-bottom: 0.2rem;
    }
    /* Section divider */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #343a40;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png",
             width=60)
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    mode = st.radio(
        "**Inference Mode**",
        ["🧠 CLIP Zero-Shot (best accuracy, any image)",
         "🚀 ViLT Demo (pretrained, any real photo)",
         "🔬 Custom Model (ResNet50+BERT or CLIP)"],
        index=0,
        help=(
            "Quick Demo (ViLT): Works on any real photo. Use this for demos.\n\n"
            "Custom Model: Educational architecture built from scratch. "
            "Trained on synthetic shapes — use with the Demo Gallery."
        ),
    )
    use_clip_zero_shot = mode.startswith("🧠")
    use_demo_model     = mode.startswith("🚀")
    use_custom_model   = mode.startswith("🔬")

    if use_custom_model:
        st.info(
            "ℹ️ Retrain with CLIP backbone for best results on real images: "
            "`python train.py --backbone clip`  (~25 min on CPU)",
        )

    st.markdown("---")
    show_attention = st.checkbox("Show attention heatmap", value=True)
    show_gradcam   = st.checkbox("Show Grad-CAM",         value=False,
                                  disabled=use_demo_model)
    show_top5      = st.checkbox("Show top-5 answers",    value=True)

    st.markdown("---")
    st.markdown("### 📚 About This Project")
    st.markdown("""
**Educational VQA Assistant**

Demonstrates multimodal AI by fusing:
- 🖼️ **ResNet-50** visual features
- 📝 **BERT** question embeddings
- 🔀 **Cross-attention** fusion

Built with PyTorch + HuggingFace Transformers.
    """)

    st.markdown("---")
    st.caption(f"Device: `{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}`")


# ─────────────────────────────────────────────────────────────
# Model loading (cached so it only loads once)
# ─────────────────────────────────────────────────────────────

# ── CLIP zero-shot answer candidates (broad real-world vocabulary) ──────
CLIP_CANDIDATES = [
    "yes", "no",
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray", "beige", "cream", "golden",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "circle", "square", "rectangle", "triangle", "oval", "round",
    "large", "small", "medium", "big", "tiny",
    "left", "right", "top", "bottom", "center", "middle",
    "wood", "metal", "plastic", "glass", "fabric", "leather", "paper",
    "indoor", "outdoor", "daytime", "nighttime", "sunny", "cloudy",
    "sitting", "standing", "walking", "running", "eating", "drinking",
    "dog", "cat", "person", "car", "tree", "building", "food", "water",
    "cup", "table", "chair", "book", "phone", "computer", "flower",
    "happy", "sad", "angry", "calm",
    "hot", "cold", "warm",
    "capacitor", "resistor", "battery", "wire",
]


@st.cache_resource(show_spinner="Loading CLIP model …")
def load_clip_zero_shot_model():
    """Load CLIP ViT-B/32 for zero-shot VQA (cached after first load)."""
    from transformers import CLIPModel, CLIPProcessor
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, processor


def predict_clip_zero_shot(clip_model, clip_processor,
                            pil_image: Image.Image,
                            question: str, top_k: int = 5) -> list[dict]:
    """
    Zero-shot VQA using CLIP image-text similarity.

    For each candidate answer we create the prompt:
        "a photo where the answer to '{question}' is {answer}"
    and score how well it matches the image.
    No task-specific training needed — works on any real image.
    """
    device = next(clip_model.parameters()).device

    texts = [f"a photo where the answer to '{question}' is {ans}"
             for ans in CLIP_CANDIDATES]

    # Process in batches of 30 to stay within memory limits on CPU
    all_scores = []
    batch_size = 30
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = clip_processor(
            text=batch,
            images=[pil_image.convert("RGB")],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clip_model(**inputs)
        # logits_per_image: (1, batch_len) — single image vs all texts in batch
        all_scores.append(out.logits_per_image[0])

    scores = torch.cat(all_scores, dim=0).softmax(dim=0)
    topk   = scores.topk(top_k)
    return [{"answer": CLIP_CANDIDATES[i.item()], "score": s.item()}
            for i, s in zip(topk.indices, topk.values)]


@st.cache_resource(show_spinner="Loading model …")
def load_demo_model():
    """Load the ViLT VQA model from HuggingFace (cached).

    Transformers v5 removed the 'visual-question-answering' pipeline task,
    so we load the processor and model directly instead.
    """
    from transformers import ViltProcessor, ViltForQuestionAnswering
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading custom model …")
def load_custom_model():
    """Load the locally trained ResNet50+BERT model (cached)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    from models.vqa_model import VQAModel
    from transformers import BertTokenizer

    ckpt = cfg.CHECKPOINT_PATH
    if not os.path.exists(ckpt):
        return None, None, None

    model, epoch, val_acc = VQAModel.load_checkpoint(ckpt, cfg.DEVICE)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)
    return model, tokenizer, val_acc


# ─────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────

def predict_demo(processor_model: tuple, pil_image: Image.Image,
                 question: str, top_k: int = 5) -> list[dict]:
    """Run ViLT inference using processor + model directly (transformers v5)."""
    processor, model = processor_model
    device = next(model.parameters()).device

    inputs = processor(pil_image.convert("RGB"), question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze(0)           # (num_labels,)
    probs  = torch.softmax(logits, dim=-1)
    topk_scores, topk_ids = probs.topk(top_k)

    return [
        {"answer": model.config.id2label[i.item()], "score": s.item()}
        for i, s in zip(topk_ids, topk_scores)
    ]


def predict_custom(model, tokenizer, pil_image: Image.Image,
                   question: str, top_k: int = 5) -> list[dict]:
    """Run custom model inference."""
    import config as cfg
    from utils.preprocessing import get_image_transforms

    transform  = get_image_transforms("val")
    img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(cfg.DEVICE)

    encoding = tokenizer(
        [question],
        max_length=cfg.MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(cfg.DEVICE)
    attention_mask = encoding["attention_mask"].to(cfg.DEVICE)

    with torch.no_grad():
        logits, img_attn, text_attn = model(img_tensor, input_ids, attention_mask)

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    topk_scores, topk_idx = probs.topk(top_k)

    answers = [
        {"answer": cfg.IDX_TO_ANSWER[i.item()], "score": s.item()}
        for i, s in zip(topk_idx, topk_scores)
    ]

    return answers, img_attn, text_attn, img_tensor.squeeze(0).cpu()


# ─────────────────────────────────────────────────────────────
# Synthetic demo gallery
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating examples …")
def get_demo_gallery(n: int = 6) -> list[dict]:
    """Generate n synthetic VQA examples for the gallery tab."""
    from data.generate_synthetic import SyntheticDataGenerator
    gen = SyntheticDataGenerator()
    return gen.generate_dataset(n)


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 Educational VQA Assistant</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Visual Question Answering with ResNet-50 + BERT + Cross-Attention</div>',
    unsafe_allow_html=True,
)

tab_predict, tab_gallery, tab_arch = st.tabs(
    ["🔍 Ask a Question", "🖼️ Demo Gallery", "🏗️ Architecture"]
)

# ─────────────────────────────────────────────────────────────
# Tab 1 — Predict
# ─────────────────────────────────────────────────────────────

with tab_predict:
    # Persist the question text across Streamlit reruns via session_state.
    # Without a key, st.text_input resets to value="" on every interaction,
    # making it impossible to type.
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">📤 Upload Image</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose an image …",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )

        # Option to use a synthetic demo image
        use_synthetic = st.checkbox("Generate a synthetic example instead", value=False)

        pil_image = None
        auto_q      = ""
        auto_answer = ""
        if use_synthetic:
            from data.generate_synthetic import SyntheticDataGenerator
            gen = SyntheticDataGenerator()
            sample = gen.generate_dataset(1)[0]
            pil_image   = sample["image"]
            auto_q      = sample["question"]
            auto_answer = sample["answer"]
            # Pre-fill the question box with the auto-generated question
            st.session_state.question_input = auto_q
            st.image(pil_image, caption="Synthetic generated image", use_container_width=True)
            st.info(f"Auto-generated question: **{auto_q}**")
        elif uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">💬 Ask a Question</div>',
                    unsafe_allow_html=True)

        # key= binds this widget to st.session_state.question_input so the
        # value survives reruns and the user can freely type and edit it.
        question = st.text_input(
            "Your question",
            key="question_input",
            placeholder="e.g. What component controls direction?",
            label_visibility="collapsed",
        )

        # Example questions — clicking one fills the box and reruns
        with st.expander("📌 Example questions"):
            example_qs = [
                "What color is the shape?",
                "What shape is the blue object?",
                "How many shapes are there?",
                "Is there a red circle in the image?",
                "Is the shape large or small?",
                "What is the color of the largest object?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(example_qs):
                if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                    st.session_state.question_input = q
                    st.rerun()

        predict_btn = st.button("🔮 Predict Answer", type="primary",
                                 use_container_width=True,
                                 disabled=pil_image is None or not question.strip())

        # ── Run inference ─────────────────────────────────────
        if predict_btn and pil_image is not None and question.strip():
            with st.spinner("Thinking …"):

                if use_clip_zero_shot:
                    # ── CLIP Zero-Shot mode ────────────────────────────
                    clip_model, clip_processor = load_clip_zero_shot_model()
                    answers = predict_clip_zero_shot(
                        clip_model, clip_processor, pil_image, question
                    )
                    best_answer = answers[0]["answer"]
                    best_score  = answers[0]["score"]

                    st.markdown(
                        f'<div class="answer-box">💡 {best_answer.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("CLIP zero-shot — no task-specific training used")

                    if show_top5:
                        st.markdown('<div class="section-title">Top-5 Answers</div>',
                                    unsafe_allow_html=True)
                        for ans in answers:
                            pct = ans["score"] * 100
                            st.markdown(f'<div class="conf-label">'
                                        f'{ans["answer"]} — {pct:.1f}%</div>',
                                        unsafe_allow_html=True)
                            st.progress(min(ans["score"] * 10, 1.0))

                elif use_demo_model:
                    # ── ViLT mode ──────────────────────────────
                    processor_model = load_demo_model()
                    answers = predict_demo(processor_model, pil_image, question)

                    best_answer = answers[0]["answer"]
                    best_score  = answers[0]["score"]

                    st.markdown(
                        f'<div class="answer-box">💡 {best_answer.upper()}</div>',
                        unsafe_allow_html=True,
                    )

                    if show_top5:
                        st.markdown('<div class="section-title">Top-5 Answers</div>',
                                    unsafe_allow_html=True)
                        for ans in answers:
                            pct = ans["score"] * 100
                            st.markdown(f'<div class="conf-label">{ans["answer"]} — {pct:.1f}%</div>',
                                        unsafe_allow_html=True)
                            st.progress(min(ans["score"], 1.0))

                else:
                    # ── Custom model mode ──────────────────────
                    model, tokenizer, val_acc = load_custom_model()

                    if model is None:
                        st.error(
                            "No trained checkpoint found. "
                            "Run  `python train.py`  first, or switch to Quick Demo mode."
                        )
                    else:
                        result = predict_custom(
                            model, tokenizer, pil_image, question
                        )
                        answers, img_attn, text_attn, img_tensor = result

                        best_answer = answers[0]["answer"]
                        best_score  = answers[0]["score"]

                        st.markdown(
                            f'<div class="answer-box">💡 {best_answer.upper()}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Checkpoint val accuracy: {val_acc*100:.1f}%")

                        if show_top5:
                            st.markdown('<div class="section-title">Top-5 Answers</div>',
                                        unsafe_allow_html=True)
                            for ans in answers:
                                pct = ans["score"] * 100
                                st.markdown(
                                    f'<div class="conf-label">{ans["answer"]} — {pct:.1f}%</div>',
                                    unsafe_allow_html=True,
                                )
                                st.progress(min(ans["score"], 1.0))

                        # ── Visualisations ─────────────────────
                        if show_attention and img_attn:
                            from utils.visualization import visualize_attention
                            import config as cfg
                            from transformers import BertTokenizer
                            tok_enc = tokenizer(question, return_tensors="pt")
                            tokens = tokenizer.convert_ids_to_tokens(
                                tok_enc["input_ids"][0]
                            )
                            # Last layer attention: (B, 49, L) → take batch 0
                            attn_w = img_attn[-1][0]  # (49, L)
                            fig = visualize_attention(img_tensor, attn_w, tokens,
                                                      title="Cross-Attention")
                            st.pyplot(fig)

                        if show_gradcam and not use_demo_model:
                            from utils.visualization import grad_cam_heatmap
                            import config as cfg
                            tok_enc = tokenizer(
                                [question], max_length=cfg.MAX_SEQ_LEN,
                                padding="max_length", truncation=True,
                                return_tensors="pt"
                            )
                            img_t = img_tensor.unsqueeze(0).to(cfg.DEVICE)
                            fig = grad_cam_heatmap(
                                model,
                                img_t,
                                tok_enc["input_ids"].to(cfg.DEVICE),
                                tok_enc["attention_mask"].to(cfg.DEVICE),
                                pil_image,
                            )
                            st.pyplot(fig)

        # Show expected answer if using synthetic sample
        if use_synthetic and not predict_btn:
            st.info(f"Expected answer: **{auto_answer}**")


# ─────────────────────────────────────────────────────────────
# Tab 2 — Demo Gallery
# ─────────────────────────────────────────────────────────────

with tab_gallery:
    st.markdown("### 🖼️ Synthetic VQA Example Gallery")
    st.markdown(
        "These are auto-generated examples showing the kinds of images and "
        "questions the model is trained on."
    )

    if st.button("🔄 Generate New Examples"):
        st.cache_data.clear()

    gallery = get_demo_gallery(n=6)

    for row_start in range(0, len(gallery), 3):
        cols = st.columns(3)
        for col_idx, sample_idx in enumerate(range(row_start, min(row_start + 3, len(gallery)))):
            s = gallery[sample_idx]
            with cols[col_idx]:
                st.image(s["image"], use_container_width=True)
                st.markdown(f"**Q:** {s['question']}")
                st.success(f"**A:** {s['answer']}")


# ─────────────────────────────────────────────────────────────
# Tab 3 — Architecture
# ─────────────────────────────────────────────────────────────

with tab_arch:
    st.markdown("### 🏗️ Model Architecture")

    st.markdown("""
```
┌─────────────────────────────────────────────────────────────────┐
│                    VQA Model Architecture                        │
├─────────────────┬─────────────────────────────────────────────── │
│                 │                                                 │
│  Image (224²)   │  Question (text)                                │
│       │         │       │                                         │
│  ┌────▼──────┐  │  ┌────▼──────┐                                  │
│  │ ResNet-50 │  │  │  BERT     │                                  │
│  │ Backbone  │  │  │  Encoder  │                                  │
│  │ (frozen)  │  │  │ (frozen)  │                                  │
│  └────┬──────┘  │  └────┬──────┘                                  │
│       │         │       │                                         │
│  (B,49,512)     │  (B,L,512)                                      │
│       │         │       │                                         │
│       └────┬────┘       │                                         │
│            │                                                      │
│   ┌─────────▼────────────┐                                        │
│   │  Cross-Attention      │  ← image tokens attend to text        │
│   │  Fusion (×2 layers)   │  ← text tokens attend to image        │
│   └─────────┬────────────┘                                        │
│             │                                                      │
│   img_fused (B,49,512)  text_fused (B,L,512)                      │
│             │                       │                              │
│        mean pool               [CLS] token                         │
│             │                       │                              │
│             └──────── cat ──────────┘                              │
│                        │                                           │
│             ┌──────────▼──────────┐                                │
│             │   MLP Classifier    │                                │
│             │  512→256→num_ans    │                                │
│             └──────────┬──────────┘                                │
│                        │                                           │
│                  Answer logits                                      │
└────────────────────────────────────────────────────────────────────┘
```
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Image Encoder (ResNet-50)**
- Pretrained on ImageNet (1.2M images)
- Extracts 49 visual tokens (7×7 spatial grid)
- Each token = 512-dim vector
- Early layers frozen, layer4 fine-tuned

**Text Encoder (BERT)**
- bert-base-uncased (12 layers, 768-dim)
- Produces contextual token embeddings
- First 10 layers frozen, last 2 fine-tuned
- Projected from 768 → 512-dim
        """)
    with col2:
        st.markdown("""
**Cross-Attention Fusion**
- 2 bidirectional cross-attention layers
- Image attends to text → which words matter for each patch
- Text attends to image → which regions support each word
- 8 attention heads, GELU activation
- Residual connections + LayerNorm

**Answer Head**
- Mean-pool image tokens + [CLS] text token
- Concatenate → 1024-dim multimodal vector
- 2-layer MLP with dropout
- Softmax over 35 answer classes
        """)

    st.markdown("---")
    st.markdown("### 🔑 Key Concepts")

    with st.expander("🔢 Embeddings"):
        st.markdown("""
An **embedding** is a dense numerical vector that represents an object
(word, image patch, answer) in a continuous vector space.

- Similar objects → similar vectors → close in the embedding space
- BERT maps words to 768-dim vectors; our model projects to 512-dim
- The projection layer creates a **shared embedding space** where image
  and text features can be directly compared
        """)

    with st.expander("🎯 Attention Mechanism"):
        st.markdown(r"""
Given Query (Q), Key (K), Value (V):

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)\!V$$

- **Q** (query): "What am I looking for?"
- **K** (key):   "What information do I have?"
- **V** (value): "What do I return when matched?"

Dividing by √d_k prevents the dot products from growing too large
(which would push softmax into saturation regions with near-zero gradients).
        """)

    with st.expander("🔀 Multimodal Fusion"):
        st.markdown("""
**Cross-Attention** is the bridge between modalities:

1. Image tokens (Q) × Text tokens (K,V)
   → Each image patch learns *which words* describe it

2. Text tokens (Q) × Image tokens (K,V)
   → Each word learns *which image regions* it refers to

After fusion, both modalities are aligned in a shared space.
The answer head reads from both to make its final prediction.
        """)

    with st.expander("🏋️ Transfer Learning"):
        st.markdown("""
Training ResNet-50 or BERT from scratch would require millions of images/texts
and weeks of GPU time.

**Transfer learning** reuses the pretrained knowledge:
- ResNet already knows edges, textures, objects (from ImageNet)
- BERT already understands English grammar and semantics

We freeze these layers and only train the new fusion layers —
much faster convergence with far less data.
        """)
