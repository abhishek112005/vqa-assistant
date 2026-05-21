"""
train.py — Training Script for the Educational VQA Model

Training Loop Overview
───────────────────────
1. Build datasets and DataLoaders.
2. Instantiate the VQA model and move it to GPU/CPU.
3. Set up the Adam optimiser with separate learning rates for:
   • pretrained backbone layers (small LR — don't overwrite ImageNet/BERT knowledge)
   • new fusion + classification layers (larger LR — learn quickly)
4. Cosine LR schedule with warm-up:
   • Warm-up: LR rises linearly for the first ~10% of steps.
     Prevents large gradient updates before the model is "warmed up".
   • Cosine decay: LR follows a cosine curve to zero.
     Smoother than step decay; tends to find better minima.
5. For each epoch:
   a. Training   — forward, loss, backward, optimiser step, LR step.
   b. Validation — forward only (no grad), compute accuracy.
   c. Save best checkpoint when val accuracy improves.

Loss function: CrossEntropyLoss
    Combines LogSoftmax + NLLLoss. Numerically stable.
    Penalises the model more when it is confidently wrong.

Gradient clipping (torch.nn.utils.clip_grad_norm_):
    Prevents "exploding gradients" — large gradient updates that can
    destabilise training, especially in Transformers.
    We clip the global gradient norm to 1.0.

Usage:
    python train.py [--epochs 15] [--batch 16] [--lr 1e-4] [--no-amp]
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from transformers import BertTokenizer
from tqdm import tqdm

import config as cfg
from models.vqa_model import VQAModel
from utils.dataset import build_dataloaders
from utils.metrics import compute_accuracy, plot_training_curves


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_param_groups(model: VQAModel, lr_backbone: float,
                     lr_fusion: float) -> list[dict]:
    """
    Separate model parameters into two groups with different learning rates.

    Why different LRs?
        Pretrained weights (ResNet, BERT) already encode rich features.
        A large LR would overwrite this knowledge ("catastrophic forgetting").
        New layers (fusion, classifier) start random → need a larger LR.
    """
    backbone_params, fusion_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder.backbone" in name or "text_encoder.bert" in name:
            backbone_params.append(param)
        else:
            fusion_params.append(param)

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": fusion_params,   "lr": lr_fusion},
    ]


def train_epoch(model, loader, optimizer, scheduler, scaler,
                criterion, device, use_amp: bool) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for batch in pbar:
        images         = batch["image"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP) uses FP16 on GPU → ~2× speed
        # Falls back to FP32 on CPU where AMP has no benefit
        with autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits, _, _ = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        # Accumulate metrics
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Val  ", leave=False, ncols=90):
        images         = batch["image"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        logits, _, _ = model(images, input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────

def main(args):
    print(f"\n{'='*60}")
    print("  Educational VQA Assistant — Training")
    print(f"{'='*60}")
    print(f"  Device  : {cfg.DEVICE}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print(f"  AMP     : {not args.no_amp}")

    # ── Tokeniser ─────────────────────────────────────────────
    print("\n[1/4] Loading tokeniser …")
    if args.backbone == "clip":
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("  Using CLIP BPE tokeniser (max_len=77)")
    else:
        tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)
        print("  Using BERT WordPiece tokeniser")

    # ── Data ──────────────────────────────────────────────────
    print("[2/4] Building datasets …")
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer, batch_size=args.batch, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────
    print("[3/4] Building model …")
    model = VQAModel(
        num_answers      = cfg.NUM_ANSWERS,
        hidden_dim       = cfg.HIDDEN_DIM,
        num_heads        = cfg.NUM_HEADS,
        num_fusion_layers= cfg.NUM_FUSION_LAYERS,
        dropout          = cfg.DROPOUT,
        pretrained       = True,
        backbone         = args.backbone,
    ).to(cfg.DEVICE)

    # Print parameter summary
    counts = model.count_parameters()
    print(f"\n  {'Module':<20} {'Total':>12}  {'Trainable':>12}")
    print(f"  {'-'*46}")
    for name, c in counts.items():
        print(f"  {name:<20} {c['total']:>12,}  {c['trainable']:>12,}")

    # ── Optimiser + Scheduler ─────────────────────────────────
    print("\n[4/4] Setting up optimiser …")
    param_groups = get_param_groups(model, cfg.LR_BACKBONE, cfg.LR_FUSION)
    optimizer = AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)

    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[cfg.LR_BACKBONE, cfg.LR_FUSION],
        total_steps=total_steps,
        pct_start=cfg.WARMUP_RATIO,   # fraction of steps for warm-up phase
        anneal_strategy="cos",         # cosine decay after warm-up
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Label smoothing: instead of target [0,0,1,0], use [ε/K,…,1-ε,…,ε/K]
    # Prevents the model from becoming overconfident → better generalisation

    scaler = GradScaler("cuda", enabled=not args.no_amp and cfg.DEVICE.type == "cuda")

    # ── Training loop ─────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    print(f"\n{'─'*60}")
    print(f"  Starting training for {args.epochs} epochs …")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{args.epochs}")

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, criterion, cfg.DEVICE, not args.no_amp
        )
        vl_loss, vl_acc = validate(model, val_loader, criterion, cfg.DEVICE)

        elapsed = time.time() - t0
        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc:.4f}   [{elapsed:.0f}s]")

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_checkpoint(cfg.CHECKPOINT_PATH, epoch,
                                   optimizer, vl_acc)

        # Always save last checkpoint
        model.save_checkpoint(cfg.CHECKPOINT_LAST_PATH, epoch,
                               optimizer, vl_acc)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Best checkpoint → {cfg.CHECKPOINT_PATH}")
    print(f"{'='*60}\n")

    # Save training curves
    fig = plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    curve_path = os.path.join(cfg.CHECKPOINT_DIR, "training_curves.png")
    fig.savefig(curve_path, dpi=120, bbox_inches="tight")
    print(f"  Training curves → {curve_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VQA model")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch",  type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=cfg.LR_FUSION)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--backbone", type=str, default="clip",
                        choices=["clip", "resnet_bert"],
                        help="clip (recommended) or resnet_bert (original)")
    args = parser.parse_args()
    main(args)
