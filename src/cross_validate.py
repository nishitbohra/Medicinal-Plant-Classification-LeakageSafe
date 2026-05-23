# src/cross_validate.py — 5-fold stratified cross-validation
# pip install torch torchvision scikit-learn scipy pandas numpy matplotlib

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import wilcoxon

from src.models import get_model
from src.dataset import get_dataloaders, compute_class_weights
from src.train import train
from src.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_kfold_cv(cfg, registry: pd.DataFrame, backbone: str, resume: bool = True) -> dict:
    """
    5-fold stratified cross-validation on ORIGINAL images only.

    For each fold:
      - Test fold: original images ONLY (no augmentation).
      - Train fold: original images + mapped augmented counterparts.
      - Trains fresh model for each fold.

    Reports per-fold and mean ± std metrics.
    Applies Wilcoxon signed-rank test vs. each baseline.
    Saves results to METRICS_DIR/cv_results_<backbone>.json.
    
    Args:
        resume: If True, will attempt to resume from checkpoint if available
    """  # PRD §14
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check for checkpoint file
    checkpoint_path = cfg.METRICS_DIR / f"cv_checkpoint_{backbone}.json"
    fold_metrics = []
    start_fold = 0
    
    if resume and checkpoint_path.exists():
        print(f"[CV] Found checkpoint at {checkpoint_path}")
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            fold_metrics = checkpoint.get("fold_metrics", [])
            start_fold = checkpoint.get("last_completed_fold", 0)
            print(f"[CV] Resuming from fold {start_fold + 1}/{cfg.CV_FOLDS}")
            print(f"[CV] {len(fold_metrics)} folds already completed\n")
        except Exception as e:
            print(f"[CV] Error loading checkpoint: {e}")
            print(f"[CV] Starting fresh...\n")
            fold_metrics = []
            start_fold = 0
    
    originals = registry[~registry["is_augmented"]].copy().reset_index(drop=True)
    augmented = registry[registry["is_augmented"]].copy()

    skf = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION — {backbone}")
    print(f"{'='*60}\n")

    for fold_idx, (train_orig_idx, test_orig_idx) in enumerate(
        skf.split(originals.index, originals["label_idx"])
    ):
        fold_num = fold_idx + 1
        
        # Skip already completed folds
        if fold_idx < start_fold:
            print(f"--- Fold {fold_num}/{cfg.CV_FOLDS} [SKIPPED - Already completed] ---")
            continue
            
        print(f"\n--- Fold {fold_num}/{cfg.CV_FOLDS} ---")

        train_orig = originals.iloc[train_orig_idx].copy()
        test_orig  = originals.iloc[test_orig_idx].copy()

        # Train paths for augmented lookup
        train_orig_stems = set(Path(p).stem for p in train_orig["path"])

        # Find augmented images whose parent original is in train fold
        aug_for_train = []
        for _, row in augmented.iterrows():
            aug_stem  = Path(row["path"]).stem
            orig_stem = re.sub(r"\s+Aug", "", aug_stem, flags=re.IGNORECASE).strip()
            # 🔴 CRITICAL: "Neem Leag Aug (N)" → "Neem Leag (N)" → normalize to "Neem Leaf (N)"
            orig_stem = orig_stem.replace("Neem Leag", "Neem Leaf")
            if orig_stem in train_orig_stems:
                aug_for_train.append(row)

        aug_train_df = pd.DataFrame(aug_for_train) if aug_for_train else pd.DataFrame(
            columns=registry.columns
        )

        # Create validation set as stratified 10% subset of train originals
        val_indices = []
        for label in train_orig["label_idx"].unique():
            label_rows = train_orig[train_orig["label_idx"] == label]
            val_size = max(1, int(len(label_rows) * 0.10))
            val_sample = label_rows.sample(n=val_size, random_state=cfg.RANDOM_SEED)
            val_indices.extend(val_sample.index.tolist())
        
        # Split train into train and val
        val_orig = train_orig.loc[val_indices].copy()
        train_orig = train_orig.loc[~train_orig.index.isin(val_indices)].copy()
        
        # Assign splits
        train_orig["split"] = "train"
        val_orig["split"] = "val" 
        test_orig["split"] = "test"

        # Prepare augmented data for training
        if len(aug_train_df) > 0:
            aug_train_df = aug_train_df.copy()
            aug_train_df["split"] = "train"

        # Combine all data for this fold - ensure all DataFrames have same columns and clean data
        fold_dfs = []
        for df in [train_orig, val_orig, test_orig]:
            df_clean = df.copy()
            # Ensure label_idx is clean integer
            df_clean["label_idx"] = df_clean["label_idx"].astype(int)
            fold_dfs.append(df_clean)
        
        if len(aug_train_df) > 0:
            aug_clean = aug_train_df.copy()
            aug_clean["label_idx"] = aug_clean["label_idx"].astype(int)
            fold_dfs.append(aug_clean)

        fold_registry = pd.concat(fold_dfs, ignore_index=True)

        loaders, datasets = get_dataloaders(fold_registry, cfg)
        class_weights     = compute_class_weights(fold_registry, cfg.NUM_CLASSES)

        model = get_model(backbone, cfg.NUM_CLASSES, cfg.PRETRAINED)
        train(cfg, model, loaders, class_weights)

        fold_result = evaluate(
            model=model,
            dataloader=loaders["test"],
            class_names=cfg.CLASS_NAMES,
            device=DEVICE,
            split_name=f"cv_fold{fold_num}",
            backbone=backbone,
            metrics_dir=cfg.METRICS_DIR,
            figures_dir=cfg.FIGURES_DIR,
        )
        fold_result["fold"] = fold_num
        fold_metrics.append(fold_result)

        print(f"[Fold {fold_num}] Macro-F1={fold_result['macro_f1']*100:.2f}%  "
              f"Acc={fold_result['accuracy']*100:.2f}%  "
              f"Kappa={fold_result['cohen_kappa']:.4f}")
        
        # Save checkpoint after each fold
        checkpoint_data = {
            "backbone": backbone,
            "last_completed_fold": fold_num,
            "fold_metrics": fold_metrics,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"[CV] Checkpoint saved (fold {fold_num} complete)")

    # Aggregate across folds
    metric_keys = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
    agg = {}
    for key in metric_keys:
        vals = [fm[key] for fm in fold_metrics]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    # Per-class F1 aggregation
    per_class_agg = {}
    for cname in cfg.CLASS_NAMES:
        vals = [fm["per_class_f1"].get(cname, 0.0) for fm in fold_metrics]
        per_class_agg[cname] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    cv_results = {
        "backbone":       backbone,
        "n_folds":        cfg.CV_FOLDS,
        "fold_metrics":   fold_metrics,
        "aggregated":     agg,
        "per_class_f1":   per_class_agg,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  CV RESULTS — {backbone}")
    print(f"{'='*60}")
    for key in metric_keys:
        m, s = agg[key]["mean"], agg[key]["std"]
        print(f"  {key:<20} {m*100:.2f}% ± {s*100:.2f}%")
    print(f"\n  Per-Class F1 (mean ± std):")
    for cname in cfg.CLASS_NAMES:
        m = per_class_agg[cname]["mean"]
        s = per_class_agg[cname]["std"]
        print(f"    {cname:<30} {m*100:.2f}% ± {s*100:.2f}%")
    print(f"{'='*60}\n")

    # Wilcoxon test vs. baselines (if CV results for baselines already saved)
    _run_wilcoxon_comparisons(cfg, cv_results, backbone)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = cfg.METRICS_DIR / f"cv_results_{backbone}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[CV] Results saved to {out_path}")
    
    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[CV] Checkpoint file removed (CV completed successfully)")

    # Plot fold-by-fold Macro-F1 bar chart
    _plot_cv_folds(fold_metrics, backbone, cfg.FIGURES_DIR)

    return cv_results


def _run_wilcoxon_comparisons(cfg, primary_cv: dict, primary_backbone: str) -> None:
    """
    Compares primary model's per-fold Macro-F1 against each baseline via Wilcoxon test.
    Loads baseline CV results from METRICS_DIR if available.
    """  # PRD §14.2
    primary_f1s = [fm["macro_f1"] for fm in primary_cv["fold_metrics"]]

    print("\n--- Wilcoxon Signed-Rank Tests vs. Baselines ---")

    for baseline in cfg.BASELINES:
        if baseline == primary_backbone:
            continue

        # Search for saved CV results file for this baseline
        baseline_files = sorted(cfg.METRICS_DIR.glob(f"cv_results_{baseline}_*.json"))
        if not baseline_files:
            print(f"  {baseline:<25} No CV results found — run CV for this baseline first.")
            continue

        with open(baseline_files[-1]) as f:   # Use most recent
            baseline_cv = json.load(f)

        baseline_f1s = [fm["macro_f1"] for fm in baseline_cv["fold_metrics"]]

        if len(primary_f1s) != len(baseline_f1s):
            print(f"  {baseline:<25} Fold count mismatch — skipping.")
            continue

        try:
            stat, p_value = wilcoxon(primary_f1s, baseline_f1s)
            sig = "✓ significant (p<0.05)" if p_value < 0.05 else "✗ not significant"
            print(f"  {primary_backbone} vs {baseline:<20} p={p_value:.4f}  {sig}")
        except Exception as e:
            print(f"  {baseline:<25} Wilcoxon error: {e}")

    print()


def _plot_cv_folds(fold_metrics: list, backbone: str, figures_dir: Path) -> None:
    """Plots fold-by-fold Macro-F1 bar chart."""
    folds   = [fm["fold"] for fm in fold_metrics]
    f1_vals = [fm["macro_f1"] * 100 for fm in fold_metrics]
    mean_f1 = np.mean(f1_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(folds, f1_vals, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axhline(mean_f1, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={mean_f1:.2f}%")

    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title(f"5-Fold CV Macro-F1 — {backbone}")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    out_path = figures_dir / f"cv_folds_{backbone}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[CV] Fold plot saved to {out_path}")
