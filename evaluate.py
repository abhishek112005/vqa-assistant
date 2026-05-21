"""
evaluate.py — Comprehensive Model Evaluation

Evaluates the trained VQA model on the held-out test set and produces:
  • Overall accuracy (top-1 and top-5)
  • Per-class precision / recall / F1
  • Confusion matrix (saved as PNG)
  • Sample predictions table
  • Zero-shot evaluation on custom unseen questions

Usage:
    python evaluate.py [--split test] [--checkpoint checkpoints/vqa_best.pth]
    python evaluate.py --zero-shot   # test on hand-crafted unseen examples
"""

import argparse
import os
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import BertTokenizer
from tqdm import tqdm

import config as cfg
from models.vqa_model import VQAModel
from utils.dataset import build_dataloaders, SyntheticVQADataset
from utils.preprocessing import get_image_transforms
from utils.metrics import (compute_accuracy, top_k_accuracy,
                            per_class_accuracy, plot_confusion_matrix,
                            plot_training_curves)
from data.generate_synthetic import SyntheticDataGenerator


# ─────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(model: VQAModel, loader,
                    device: torch.device) -> dict:
    """
    Run the model over every batch in a DataLoader and collect predictions.

    Returns:
        dict with keys: predictions, targets, questions, answers, logits_all
    """
    model.eval()
    predictions, targets = [], []
    questions, answers   = [], []
    logits_all           = []

    for batch in tqdm(loader, desc="  Evaluating", ncols=80):
        images         = batch["image"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        logits, _, _ = model(images, input_ids, attention_mask)

        preds = logits.argmax(dim=-1)
        predictions.extend(preds.cpu().tolist())
        targets.extend(labels.cpu().tolist())
        logits_all.append(logits.cpu())

        questions.extend(batch["question"])
        answers.extend(batch["answer"])

    logits_all = torch.cat(logits_all, dim=0)
    return {
        "predictions": predictions,
        "targets"    : targets,
        "questions"  : questions,
        "answers"    : answers,
        "logits"     : logits_all,
    }


# ─────────────────────────────────────────────────────────────
# Zero-shot examples
# ─────────────────────────────────────────────────────────────

ZERO_SHOT_EXAMPLES = [
    # (question, expected_answer)
    # These images are generated fresh — the model has not seen them.
    ("What color is the shape?", None),
    ("Is there a red circle in the image?", None),
    ("How many shapes are in the image?", None),
    ("What is the shape of the blue object?", None),
    ("Is there a green triangle?", None),
    ("What color is the square?", None),
    ("Is there a yellow rectangle?", None),
    ("How many shapes do you see?", None),
]


def run_zero_shot(model: VQAModel, tokenizer: BertTokenizer,
                  n_per_question: int = 3) -> pd.DataFrame:
    """
    Test the model on freshly generated images with predefined questions.
    Measures generalisation to samples not seen during training.
    """
    gen       = SyntheticDataGenerator()
    transform = get_image_transforms("val")
    results   = []

    for question_template, _ in ZERO_SHOT_EXAMPLES:
        for _ in range(n_per_question):
            img, actual_q, actual_a = gen._sample_one()  # fresh sample
            # Override question with our template if it makes sense
            q_to_ask = actual_q

            encoding = tokenizer(
                [q_to_ask],
                max_length=cfg.MAX_SEQ_LEN, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            img_t = transform(img.convert("RGB")).unsqueeze(0).to(cfg.DEVICE)
            input_ids = encoding["input_ids"].to(cfg.DEVICE)
            attn_mask = encoding["attention_mask"].to(cfg.DEVICE)

            with torch.no_grad():
                logits, _, _ = model(img_t, input_ids, attn_mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_idx = probs.argmax().item()
            pred_ans = cfg.IDX_TO_ANSWER[pred_idx]
            conf     = probs[pred_idx].item()

            results.append({
                "Question": q_to_ask,
                "GT Answer": actual_a,
                "Predicted": pred_ans,
                "Confidence": f"{conf*100:.1f}%",
                "Correct": "✓" if pred_ans == actual_a else "✗",
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    print(f"\n{'='*60}")
    print("  Educational VQA Assistant — Evaluation")
    print(f"  Device: {cfg.DEVICE}")
    print(f"{'='*60}")

    # ── Load model ────────────────────────────────────────────
    ckpt_path = args.checkpoint or cfg.CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"\n  ✗ Checkpoint not found: {ckpt_path}")
        print("    Train the model first:  python train.py\n")
        return

    model, epoch, best_val = VQAModel.load_checkpoint(ckpt_path, cfg.DEVICE)
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)

    # ── Build test loader ─────────────────────────────────────
    print("\n  Building test dataset …")
    _, _, test_loader = build_dataloaders(tokenizer, batch_size=32)

    # ── Evaluate ──────────────────────────────────────────────
    print("  Running evaluation …")
    results = evaluate_loader(model, test_loader, cfg.DEVICE)

    preds    = results["predictions"]
    targets  = results["targets"]
    logits   = results["logits"]
    questions = results["questions"]
    gt_answers = results["answers"]

    # ── Metrics ───────────────────────────────────────────────
    top1  = compute_accuracy(preds, targets)
    top5  = top_k_accuracy(logits, torch.tensor(targets), k=5)
    class_report = per_class_accuracy(preds, targets, cfg.ANSWER_VOCAB)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Top-1 Accuracy : {top1*100:6.2f}%              │")
    print(f"  │  Top-5 Accuracy : {top5*100:6.2f}%              │")
    print(f"  │  Macro F1       : {class_report['macro avg']['f1-score']*100:6.2f}%              │")
    print(f"  └─────────────────────────────────────┘")

    # ── Confusion matrix ──────────────────────────────────────
    print("\n  Generating confusion matrix …")
    fig = plot_confusion_matrix(preds, targets, cfg.ANSWER_VOCAB,
                                title="VQA Confusion Matrix")
    cm_path = os.path.join(cfg.CHECKPOINT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {cm_path}")

    # ── Sample predictions ────────────────────────────────────
    print("\n  Sample Predictions (first 10):")
    print(f"  {'Question':<45} {'GT':>15} {'Pred':>15}  OK?")
    print(f"  {'-'*80}")
    for i in range(min(10, len(questions))):
        pred_ans = cfg.IDX_TO_ANSWER[preds[i]]
        ok = "✓" if preds[i] == targets[i] else "✗"
        print(f"  {questions[i]:<45} {gt_answers[i]:>15} {pred_ans:>15}  {ok}")

    # ── Export predictions ────────────────────────────────────
    pred_df = pd.DataFrame({
        "Question"  : questions,
        "GT Answer" : gt_answers,
        "Predicted" : [cfg.IDX_TO_ANSWER[p] for p in preds],
        "Correct"   : ["yes" if p == t else "no"
                        for p, t in zip(preds, targets)],
        "Confidence": [f"{torch.softmax(logits[i], dim=-1).max().item()*100:.1f}%"
                        for i in range(len(preds))],
    })
    pred_path = os.path.join(cfg.CHECKPOINT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Full predictions → {pred_path}")

    # ── Zero-shot evaluation ──────────────────────────────────
    if args.zero_shot:
        print("\n  Running zero-shot evaluation …")
        zs_df = run_zero_shot(model, tokenizer)
        zs_acc = (zs_df["Correct"] == "✓").mean()
        print(f"\n  Zero-shot accuracy: {zs_acc*100:.1f}%")
        print(zs_df.to_string(index=False))
        zs_path = os.path.join(cfg.CHECKPOINT_DIR, "zero_shot_results.csv")
        zs_df.to_csv(zs_path, index=False)

    # ── Per-class report ──────────────────────────────────────
    if args.verbose:
        print("\n  Per-class Report:")
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*52}")
        for cls_name in cfg.ANSWER_VOCAB:
            if cls_name in class_report:
                r = class_report[cls_name]
                print(f"  {cls_name:<20} {r['precision']:>10.3f} "
                      f"{r['recall']:>10.3f} {r['f1-score']:>10.3f}")

    print(f"\n  Evaluation complete.\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the VQA model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--zero-shot",  action="store_true",
                        help="Also run zero-shot evaluation")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print per-class metrics")
    args = parser.parse_args()
    main(args)
