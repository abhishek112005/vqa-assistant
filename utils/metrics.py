"""
utils/metrics.py — Evaluation Metrics for VQA

VQA Accuracy (the official metric from the VQA challenge):
    Rather than requiring an exact match, the VQA dataset collects
    10 human annotations per question and computes:
        min(count_of_predicted_answer / 3, 1.0)
    This gives partial credit if most but not all annotators agree.
    For our single-annotator synthetic dataset we use exact match,
    but the vqa_score() function is included for reference.

Top-K Accuracy:
    The model is correct if the right answer appears in its top-K
    predictions.  Top-5 is typically much higher than Top-1 and shows
    the model "knows" the answer even if ranking is imperfect.

Confusion Matrix:
    Cell (i, j) = number of times class i was predicted as class j.
    Diagonal = correct predictions. Off-diagonal = errors.
    Reveals which answer classes are most confused with each other.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")        # non-interactive backend safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, classification_report


def compute_accuracy(predictions: list[int] | torch.Tensor,
                     targets: list[int] | torch.Tensor) -> float:
    """Exact-match top-1 accuracy (0.0–1.0)."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = np.array(predictions)
    targets     = np.array(targets)
    return float((predictions == targets).mean())


def top_k_accuracy(logits: torch.Tensor,
                   targets: torch.Tensor, k: int = 5) -> float:
    """
    Top-K accuracy: fraction of samples where the true class
    is among the model's K highest-scored answers.
    """
    with torch.no_grad():
        topk = logits.topk(min(k, logits.size(-1)), dim=-1).indices  # (B, k)
        correct = topk.eq(targets.unsqueeze(1).expand_as(topk))      # (B, k)
        return float(correct.any(dim=1).float().mean().item())


def vqa_score(predicted_answer: str, ground_truth_answers: list[str]) -> float:
    """
    Official VQA soft-accuracy for a single example.
    Returns min(count of predicted in GT list / 3, 1.0).
    With a single GT label (our synthetic set), this equals 0 or 0.33.
    """
    count = ground_truth_answers.count(predicted_answer)
    return min(count / 3.0, 1.0)


def per_class_accuracy(predictions: list[int],
                       targets: list[int],
                       class_names: list[str] = None) -> dict:
    """Return per-class precision, recall, F1 as a dict."""
    report = classification_report(
        targets, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def plot_confusion_matrix(predictions: list[int],
                          targets: list[int],
                          class_names: list[str],
                          title: str = "Confusion Matrix",
                          figsize: tuple = (14, 12)) -> plt.Figure:
    """
    Plot and return a confusion matrix figure.

    Only classes that appear in the data are shown (avoids empty rows/cols).
    Cell values are normalised by true class count → recall per class.
    """
    # Keep only classes that actually appear
    present = sorted(set(targets) | set(predictions))
    labels  = [class_names[i] for i in present]

    cm = confusion_matrix(targets, predictions, labels=present)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)

    # Tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells with raw counts
    thresh = cm_norm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_training_curves(train_losses: list[float],
                         val_losses: list[float],
                         train_accs: list[float],
                         val_accs: list[float]) -> plt.Figure:
    """Plot loss and accuracy curves side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Loss Curves",     fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_accs], "b-o",
             label="Train Acc", markersize=4)
    ax2.plot(epochs, [a * 100 for a in val_accs],   "r-o",
             label="Val Acc",   markersize=4)
    ax2.set_title("Accuracy Curves",  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig
