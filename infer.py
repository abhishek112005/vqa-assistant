"""
infer.py — Single-Sample Inference

Run the trained VQA model on one image + question and print the answer
with a confidence score and top-5 predictions.

Usage:
    python infer.py --image path/to/image.png --question "What color is the circle?"
    python infer.py --demo          # run on built-in synthetic examples
"""

import argparse
import os
import random
import sys

import torch
from PIL import Image
from transformers import BertTokenizer

import config as cfg
from models.vqa_model import VQAModel
from utils.preprocessing import load_image, get_image_transforms
from data.generate_synthetic import SyntheticDataGenerator


def load_model(checkpoint_path: str = None) -> tuple[VQAModel, BertTokenizer]:
    """Load the trained model and tokeniser from a checkpoint."""
    path = checkpoint_path or cfg.CHECKPOINT_PATH

    if not os.path.exists(path):
        print(f"\n  ✗ Checkpoint not found: {path}")
        print("    Run  python train.py  first to train the model.\n")
        sys.exit(1)

    print(f"  Loading checkpoint: {path}")
    model, epoch, val_acc = VQAModel.load_checkpoint(path, cfg.DEVICE)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)
    return model, tokenizer


def predict(model: VQAModel, tokenizer: BertTokenizer,
            pil_image: Image.Image, question: str,
            top_k: int = 5) -> dict:
    """
    Run inference on one image + question.

    Args:
        model     : trained VQAModel in eval mode
        tokenizer : matching BertTokenizer
        pil_image : PIL image (any size / mode)
        question  : natural language question string
        top_k     : number of top answers to return

    Returns:
        dict with keys: answer, confidence, top_k_answers, top_k_scores
    """
    # ── Preprocess ────────────────────────────────────────────
    transform   = get_image_transforms("val")
    img_tensor  = transform(pil_image.convert("RGB")).unsqueeze(0).to(cfg.DEVICE)

    encoding = tokenizer(
        [question],
        max_length=cfg.MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(cfg.DEVICE)
    attention_mask = encoding["attention_mask"].to(cfg.DEVICE)

    # ── Forward pass ──────────────────────────────────────────
    with torch.no_grad():
        logits, img_attn, text_attn = model(img_tensor, input_ids, attention_mask)

    probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_answers,)

    # Top-K answers
    topk_scores, topk_indices = probs.topk(top_k)
    topk_answers = [cfg.IDX_TO_ANSWER[i.item()] for i in topk_indices]
    topk_scores  = topk_scores.cpu().tolist()

    return {
        "answer"       : topk_answers[0],
        "confidence"   : topk_scores[0],
        "top_k_answers": topk_answers,
        "top_k_scores" : topk_scores,
        "img_attn"     : img_attn,
        "text_attn"    : text_attn,
        "img_tensor"   : img_tensor.squeeze(0).cpu(),
    }


def print_result(question: str, result: dict, expected: str = None):
    """Pretty-print the prediction result."""
    bar = "─" * 50
    print(f"\n  {bar}")
    print(f"  Question  : {question}")
    print(f"  Answer    : {result['answer']}  "
          f"(confidence {result['confidence']*100:.1f}%)")
    if expected is not None:
        match = "✓" if result["answer"] == expected else "✗"
        print(f"  Expected  : {expected}  {match}")
    print(f"  Top-5     :")
    for ans, sc in zip(result["top_k_answers"], result["top_k_scores"]):
        bar_len = int(sc * 30)
        print(f"    {'█' * bar_len:<30}  {ans:<15} {sc*100:5.1f}%")
    print(f"  {bar}")


# ─────────────────────────────────────────────────────────────
# Demo mode — run on synthetic samples
# ─────────────────────────────────────────────────────────────

def run_demo(model: VQAModel, tokenizer: BertTokenizer, n: int = 5):
    """Generate and evaluate n synthetic samples."""
    gen = SyntheticDataGenerator()
    samples = gen.generate_dataset(n)

    correct = 0
    for i, s in enumerate(samples, 1):
        print(f"\n  ── Sample {i}/{n} ──────────────────────────────")
        result = predict(model, tokenizer, s["image"], s["question"])
        print_result(s["question"], result, expected=s["answer"])
        if result["answer"] == s["answer"]:
            correct += 1

    print(f"\n  Demo accuracy: {correct}/{n}  ({correct/n*100:.0f}%)\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Inference")
    parser.add_argument("--image",    type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--question", type=str, default=None,
                        help="Question about the image")
    parser.add_argument("--demo",     action="store_true",
                        help="Run on synthetic demo samples")
    parser.add_argument("--n",        type=int, default=5,
                        help="Number of demo samples")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best checkpoint)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Educational VQA Assistant — Inference")
    print(f"  Device: {cfg.DEVICE}")
    print(f"{'='*55}")

    model, tokenizer = load_model(args.checkpoint)

    if args.demo:
        run_demo(model, tokenizer, n=args.n)
    elif args.image and args.question:
        pil_image = Image.open(args.image).convert("RGB")
        result = predict(model, tokenizer, pil_image, args.question)
        print_result(args.question, result)
    else:
        print("\n  Specify --image and --question, or use --demo\n")
        parser.print_help()
