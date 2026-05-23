# src/evaluate.py — All evaluation metrics + CI + confusion matrix
# pip install torch scikit-learn seaborn matplotlib pandas numpy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap: int = 1000,
                  alpha: float = 0.05, seed: int = 42) -> tuple:
    """
    Computes bootstrap confidence interval for a scalar metric.
    Returns (lower_bound, upper_bound) at (1-alpha) confidence level.
    """
    rng    = np.random.RandomState(seed)
    scores = []
    n      = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true_arr[idx], y_pred_arr[idx])
            scores.append(score)
        except Exception:
            pass   # Skip degenerate bootstrap samples (e.g., single class)

    if not scores:
        return 0.0, 0.0
    lower = float(np.percentile(scores, 100 * (alpha / 2)))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper


def evaluate(
    model: torch.nn.Module,
    dataloader,
    class_names: list,
    device,
    split_name: str,
    backbone: str,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict:
    """
    Full evaluation on a dataloader. Returns metrics dict.

    Computes:
      - Macro-F1 (PRIMARY)
      - Overall Accuracy
      - Per-class Accuracy
      - Weighted-F1, Macro Precision, Macro Recall
      - Cohen's Kappa
      - ROC-AUC per class (One-vs-Rest)
      - Macro ROC-AUC
      - Confusion Matrix (raw + normalized)
      - Bootstrap 95% CIs for Macro-F1, Accuracy, Cohen's Kappa
    """  # PRD §9.1
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []   # Softmax probabilities for ROC-AUC

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    num_classes = len(class_names)

    # --- Core metrics ---
    accuracy       = accuracy_score(all_labels, all_preds)
    macro_f1       = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1    = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_prec     = precision_score(all_labels, all_preds, average="macro",    zero_division=0)
    macro_recall   = recall_score(all_labels, all_preds, average="macro",       zero_division=0)
    cohen_kappa    = cohen_kappa_score(all_labels, all_preds)
    per_class_f1   = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Per-class accuracy
    per_class_acc = {}
    for c, cname in enumerate(class_names):
        mask = (all_labels == c)
        if mask.sum() > 0:
            per_class_acc[cname] = float(accuracy_score(all_labels[mask], all_preds[mask]))
        else:
            per_class_acc[cname] = 0.0

    # --- ROC-AUC (One-vs-Rest) ---
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    per_class_auc = {}
    for c, cname in enumerate(class_names):
        try:
            auc = roc_auc_score(labels_bin[:, c], all_probs[:, c])
        except ValueError:
            auc = 0.0
        per_class_auc[cname] = float(auc)

    try:
        macro_auc = roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        macro_auc = 0.0

    # --- Bootstrap CIs ---
    def _macro_f1_fn(yt, yp):
        return f1_score(yt, yp, average="macro", zero_division=0)

    def _kappa_fn(yt, yp):
        return cohen_kappa_score(yt, yp)

    ci_macro_f1  = _bootstrap_ci(all_labels, all_preds, _macro_f1_fn)
    ci_accuracy  = _bootstrap_ci(all_labels, all_preds, accuracy_score)
    ci_kappa     = _bootstrap_ci(all_labels, all_preds, _kappa_fn)

    # --- Confusion matrix ---
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Build metrics dict
    neem_idx  = class_names.index("Neem Leaf") if "Neem Leaf" in class_names else -1
    neem_f1   = float(per_class_f1[neem_idx]) if neem_idx >= 0 else 0.0
    neem_test_count = int((all_labels == neem_idx).sum()) if neem_idx >= 0 else 0

    metrics = {
        "backbone":          backbone,
        "split":             split_name,
        "accuracy":          float(accuracy),
        "macro_f1":          float(macro_f1),
        "weighted_f1":       float(weighted_f1),
        "macro_precision":   float(macro_prec),
        "macro_recall":      float(macro_recall),
        "cohen_kappa":       float(cohen_kappa),
        "macro_auc":         float(macro_auc),
        "per_class_f1":      {cn: float(per_class_f1[i]) for i, cn in enumerate(class_names)},
        "per_class_acc":     per_class_acc,
        "per_class_auc":     per_class_auc,
        "ci_macro_f1":       list(ci_macro_f1),
        "ci_accuracy":       list(ci_accuracy),
        "ci_kappa":          list(ci_kappa),
        "neem_test_count":   neem_test_count,
        "neem_f1":           neem_f1,
        "confusion_matrix":  cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }

    # --- Save metrics JSON ---
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path  = metrics_dir / f"{split_name}_metrics_{backbone}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Plot confusion matrices ---
    _plot_confusion_matrix(cm, cm_norm, class_names, backbone, split_name, figures_dir)

    # --- Plot ROC curves ---
    _plot_roc_curves(all_labels, all_probs, class_names, num_classes,
                     backbone, split_name, figures_dir)

    # --- Print formatted table to stdout ---
    _print_metrics_table(metrics, class_names)

    print(f"\n[Evaluate] Metrics saved to {json_path}")
    return metrics


def _print_metrics_table(metrics: dict, class_names: list) -> None:
    """Prints paper-ready metrics table to stdout."""  # PRD §10
    b  = metrics["backbone"]
    sp = metrics["split"].upper()
    ci_f1  = metrics["ci_macro_f1"]
    ci_acc = metrics["ci_accuracy"]
    ci_k   = metrics["ci_kappa"]
    nc     = metrics.get("neem_test_count", 0)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {b} — {sp} SET")
    print(f"{'='*60}")
    print(f"Macro-F1 Score (PRIMARY):     {metrics['macro_f1']*100:.2f}%  "
          f"[95% CI: {ci_f1[0]*100:.2f}–{ci_f1[1]*100:.2f}%]")
    print(f"Overall Accuracy:             {metrics['accuracy']*100:.2f}%  "
          f"[95% CI: {ci_acc[0]*100:.2f}–{ci_acc[1]*100:.2f}%]")
    print(f"Weighted-F1:                  {metrics['weighted_f1']*100:.2f}%")
    print(f"Macro Precision:              {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:                 {metrics['macro_recall']*100:.2f}%")
    print(f"Cohen's Kappa (κ):            {metrics['cohen_kappa']:.4f}  "
          f"[95% CI: {ci_k[0]:.4f}–{ci_k[1]:.4f}]")
    print(f"Macro ROC-AUC:                {metrics['macro_auc']:.4f}")
    print()
    print("Per-Class F1:")
    for cname in class_names:
        f1_val = metrics["per_class_f1"].get(cname, 0.0)
        suffix = ""
        if cname == "Neem Leaf":
            suffix = f"  ⚠ ({nc} test samples — CI spans ±31%)"
        print(f"  {cname:<30} {f1_val*100:.2f}%{suffix}")
    print(f"{'='*60}\n")


def _plot_confusion_matrix(
    cm: np.ndarray,
    cm_norm: np.ndarray,
    class_names: list,
    backbone: str,
    split_name: str,
    figures_dir: Path,
) -> None:
    """Plots raw + normalized confusion matrices as seaborn heatmaps."""
    short_names = [c.replace(" Leaf", "") for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title(f"Confusion Matrix (Counts)\n{backbone} — {split_name}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized (percentages) — annotate with both % and count
    annot_text = np.array([
        [f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])
    sns.heatmap(
        cm_norm, annot=annot_text, fmt="", cmap="Oranges",
        xticklabels=short_names, yticklabels=short_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
    )
    axes[1].set_title(f"Confusion Matrix (Normalized)\n{backbone} — {split_name}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    out_path = figures_dir / f"confusion_matrix_{split_name}_{backbone}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to {out_path}")


def _plot_roc_curves(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    class_names: list,
    num_classes: int,
    backbone: str,
    split_name: str,
    figures_dir: Path,
) -> None:
    """Plots per-class and macro-average ROC curves."""
    from sklearn.metrics import roc_curve, auc as sklearn_auc

    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    tpr_macro = []
    mean_fpr  = np.linspace(0, 1, 200)

    for c, (cname, color) in enumerate(zip(class_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(labels_bin[:, c], all_probs[:, c])
            roc_auc     = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{cname} (AUC={roc_auc:.3f})")
            tpr_macro.append(np.interp(mean_fpr, fpr, tpr))
        except ValueError:
            pass

    # Macro-average
    if tpr_macro:
        mean_tpr = np.mean(tpr_macro, axis=0)
        macro_auc_val = sklearn_auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
                label=f"Macro Average (AUC={macro_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {backbone} — {split_name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = figures_dir / f"roc_curves_{split_name}_{backbone}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] ROC curves saved to {out_path}")
