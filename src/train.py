# src/train.py — Training loop with class-weighted loss
# pip install torch torchvision scikit-learn matplotlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import copy
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from src.models import unfreeze_backbone

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_val_metrics(model: nn.Module, loader, criterion, device) -> dict:
    """Runs inference on val/test loader; returns loss, accuracy, macro-F1, weighted-F1."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            n_batches  += 1

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss      = total_loss / max(n_batches, 1)
    accuracy      = accuracy_score(all_labels, all_preds)
    macro_f1      = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss":        avg_loss,
        "accuracy":    accuracy,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
    }


def train(cfg, model: nn.Module, dataloaders: dict, class_weights: torch.Tensor, start_epoch: int = 1, logger=None) -> dict:
    """
    Full training loop. Returns history dict with all metrics per epoch.

    Loss:      CrossEntropyLoss(weight=class_weights)  # PRD §6.1
    Optimizer: AdamW(lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    Scheduler: CosineAnnealingWarmRestarts(T_0=cfg.T_0)

    Early stopping monitors val_macro_f1 (NOT accuracy).  # PRD §6.2
    Best checkpoint saved to CHECKPOINT_DIR/<backbone>_best.pth.
    Backbone unfrozen after epoch 5.
    """
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.T_0
    )

    history = {
        "train_loss":          [],
        "val_loss":            [],
        "val_accuracy":        [],
        "val_macro_f1":        [],
        "val_weighted_f1":     [],
        "lr":                  [],
    }

    best_macro_f1   = -1.0
    patience_count  = 0
    best_state_dict = None
    backbone        = cfg.BACKBONE

    print(f"\n{'='*60}")
    print(f"  TRAINING — {backbone} on {DEVICE}")
    print(f"  Epochs={cfg.EPOCHS}  BS={cfg.BATCH_SIZE}  LR={cfg.LEARNING_RATE}")
    print(f"  Early stop patience={cfg.EARLY_STOP_PAT} (monitoring val_macro_f1)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        epoch_start = time.time()

        # --- Unfreeze backbone after warm-up ---
        if epoch == 6:   # After epoch 5 completes, i.e., at the start of epoch 6
            print(f"\n[Epoch {epoch}] Unfreezing backbone for fine-tuning...")
            if logger:
                logger.info(f"Unfreezing backbone at epoch {epoch}")
            unfreeze_backbone(model, backbone)
            # Re-create optimizer to include newly unfrozen parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.LEARNING_RATE * 0.1,   # Lower LR for fine-tuning
                weight_decay=cfg.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0
            )

        # --- Training phase ---
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for images, labels in dataloaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss      += loss.item()
            n_train_batches += 1

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_train_loss = train_loss / max(n_train_batches, 1)

        # --- Validation phase ---
        val_metrics = _compute_val_metrics(model, dataloaders["val"], criterion, DEVICE)

        # --- Record history ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:>3}/{cfg.EPOCHS}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']*100:.2f}%  "
            f"val_macro_f1={val_metrics['macro_f1']*100:.2f}%  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # --- Early stopping on val_macro_f1  # PRD §6.2 ---
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1   = val_metrics["macro_f1"]
            patience_count  = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            # Save checkpoint
            ckpt_path = cfg.CHECKPOINT_DIR / f"{backbone}_best.pth"
            torch.save({
                "model_state_dict": best_state_dict,
                "epoch":            epoch,
                "val_macro_f1":     best_macro_f1,
                "backbone":         backbone,
                "cfg":              {
                    "BACKBONE":       cfg.BACKBONE,
                    "NUM_CLASSES":    cfg.NUM_CLASSES,
                    "INPUT_SIZE":     cfg.INPUT_SIZE,
                    "LEARNING_RATE":  cfg.LEARNING_RATE,
                    "RANDOM_SEED":    cfg.RANDOM_SEED,
                },
            }, ckpt_path)
            print(f"  ✓ Best checkpoint saved (macro_f1={best_macro_f1*100:.2f}%) → {ckpt_path}")
            if logger:
                logger.info(f"New best model saved: epoch {epoch}, macro_f1={best_macro_f1*100:.2f}%")

        else:
            patience_count += 1
            if patience_count >= cfg.EARLY_STOP_PAT:
                early_stop_msg = (f"\n[Early Stop] No improvement for {cfg.EARLY_STOP_PAT} epochs. "
                                f"Best val_macro_f1={best_macro_f1*100:.2f}%")
                print(early_stop_msg)
                if logger:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save training history
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = cfg.METRICS_DIR / f"training_history_{backbone}_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved to {history_path}")

    # Plot learning curves
    _plot_training_curves(history, backbone, cfg.FIGURES_DIR)

    print(f"\n[Train] COMPLETE — Best val_macro_f1={best_macro_f1*100:.2f}%")
    return history


def _plot_training_curves(history: dict, backbone: str, figures_dir: Path) -> None:
    """Plots loss and macro-F1 curves, saves to figures_dir/training_curves_<backbone>.png."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="darkorange")
    axes[0].set_title(f"Loss Curves — {backbone}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Macro-F1 curve
    axes[1].plot(epochs, [v * 100 for v in history["val_macro_f1"]],
                 label="Val Macro-F1", color="green")
    axes[1].set_title(f"Macro-F1 — {backbone}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / f"training_curves_{backbone}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Train] Curves saved to {out_path}")
