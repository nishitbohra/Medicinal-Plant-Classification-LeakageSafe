# src/ablation.py — Ablation experiments E1–E5
# pip install torch torchvision scikit-learn pandas numpy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models import get_model
from src.dataset import get_dataloaders, compute_class_weights, get_transforms, MedicinalLeafDataset
from src.train import train
from src.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_ablation_completed(ablation_id: str, metrics_dir: Path) -> bool:
    """
    Checks if an ablation experiment has already been completed.
    Returns True if a result file exists for this experiment.
    """
    # Look for any file matching the pattern
    pattern = f"ablation_{ablation_id}_*.json"
    existing_files = list(metrics_dir.glob(pattern))
    
    if existing_files:
        print(f"[Ablation {ablation_id}] ✓ Already completed. Found: {existing_files[0].name}")
        return True
    return False


def _load_ablation_result(ablation_id: str, metrics_dir: Path) -> dict:
    """
    Loads the most recent ablation result for the given experiment ID.
    Returns the loaded dict or None if not found.
    """
    pattern = f"ablation_{ablation_id}_*.json"
    existing_files = sorted(metrics_dir.glob(pattern), reverse=True)
    
    if existing_files:
        with open(existing_files[0], "r") as f:
            result = json.load(f)
        print(f"[Ablation {ablation_id}] Loaded existing results from {existing_files[0].name}")
        return result
    return None


def _save_ablation_result(result: dict, ablation_id: str, metrics_dir: Path) -> None:
    """Saves ablation result dict to JSON with timestamp."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = metrics_dir / f"ablation_{ablation_id}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Ablation {ablation_id}] Results saved to {out_path}")


def _quick_train_eval(cfg, registry: pd.DataFrame, train_mask: pd.Series,
                      tag: str) -> dict:
    """
    Trains model on registry[train_mask] and evaluates on test split.
    Returns metrics dict.
    """
    sub_registry = registry.copy()
    # Mark non-selected rows as excluded
    sub_registry.loc[~train_mask & (sub_registry["split"] == "train"), "split"] = None
    sub_registry = sub_registry[sub_registry["split"].notna()].copy()

    loaders, datasets = get_dataloaders(sub_registry, cfg)
    class_weights     = compute_class_weights(sub_registry, cfg.NUM_CLASSES)

    model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
    train(cfg, model, loaders, class_weights)

    test_metrics = evaluate(
        model=model,
        dataloader=loaders["test"],
        class_names=cfg.CLASS_NAMES,
        device=DEVICE,
        split_name=f"test_{tag}",
        backbone=cfg.BACKBONE,
        metrics_dir=cfg.METRICS_DIR,
        figures_dir=cfg.FIGURES_DIR,
    )
    return test_metrics


def experiment_e1_augmentation_value(cfg, registry: pd.DataFrame) -> dict:
    """
    E1: Augmentation value — originals only vs. originals + augmented.
    Quantifies augmentation benefit, especially Neem recall.
    Trains two identical models; reports Macro-F1 and Neem recall for both.
    """
    print("\n" + "="*60)
    print("  ABLATION E1 — AUGMENTATION VALUE")
    print("="*60)
    
    # Check if already completed
    if _check_ablation_completed("e1_augmentation", cfg.METRICS_DIR):
        return _load_ablation_result("e1_augmentation", cfg.METRICS_DIR)

    # Condition A: originals only (no augmented images in train)
    no_aug_registry = registry[
        ((registry["split"] == "train") & (~registry["is_augmented"])) |
        (registry["split"].isin(["val", "test"]))
    ].copy()

    # Condition B: originals + augmented (standard protocol)
    with_aug_registry = registry[registry["split"].notna()].copy()

    results = {}

    for condition, reg in [("no_augmentation", no_aug_registry),
                            ("with_augmentation", with_aug_registry)]:
        print(f"\n--- E1: Condition '{condition}' ---")
        loaders, _ = get_dataloaders(reg, cfg)
        class_weights = compute_class_weights(reg, cfg.NUM_CLASSES)
        model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
        train(cfg, model, loaders, class_weights)
        metrics = evaluate(
            model=model,
            dataloader=loaders["test"],
            class_names=cfg.CLASS_NAMES,
            device=DEVICE,
            split_name=f"test_e1_{condition}",
            backbone=cfg.BACKBONE,
            metrics_dir=cfg.METRICS_DIR,
            figures_dir=cfg.FIGURES_DIR,
        )
        neem_recall = metrics["per_class_f1"].get("Neem Leaf", 0.0)
        results[condition] = {
            "macro_f1":    metrics["macro_f1"],
            "accuracy":    metrics["accuracy"],
            "neem_recall": neem_recall,
        }

    delta_f1 = results["with_augmentation"]["macro_f1"] - results["no_augmentation"]["macro_f1"]
    results["delta_macro_f1"] = delta_f1

    print(f"\n[E1] No aug: macro_f1={results['no_augmentation']['macro_f1']*100:.2f}%  "
          f"Neem F1={results['no_augmentation']['neem_recall']*100:.2f}%")
    print(f"[E1] With aug: macro_f1={results['with_augmentation']['macro_f1']*100:.2f}%  "
          f"Neem F1={results['with_augmentation']['neem_recall']*100:.2f}%")
    print(f"[E1] Delta Macro-F1 from augmentation: +{delta_f1*100:.2f} pp")

    _save_ablation_result(results, "e1_augmentation", cfg.METRICS_DIR)
    return results


def experiment_e2_leakage_contamination(cfg, registry: pd.DataFrame) -> dict:
    """
    E2: Leakage-contaminated split vs. leakage-safe split.

    Contaminated: combine ALL images (orig + aug) → random 80/20 split.
    Safe: split on originals only (standard protocol).

    PRIMARY methodological contribution of this paper.
    Expected: contaminated accuracy is 10–25% HIGHER than leakage-safe.
    """  # PRD §12 — 🔴 CRITICAL: Primary paper contribution
    print("\n" + "="*60)
    print("  ABLATION E2 — LEAKAGE CONTAMINATION IMPACT")
    print("="*60 + "\n")
    
    # Check if already completed
    if _check_ablation_completed("e2_leakage", cfg.METRICS_DIR):
        return _load_ablation_result("e2_leakage", cfg.METRICS_DIR)

    results = {}

    # --- Condition A: Leakage-contaminated split ---
    print("--- E2: Condition 'contaminated' (ALL images → random 80/20) ---")

    all_images    = registry.copy()
    all_labels    = all_images["label_idx"].values
    n_total       = len(all_images)

    contaminated_train_idx, contaminated_test_idx = train_test_split(
        np.arange(n_total),
        test_size=0.20,
        stratify=all_labels,
        random_state=cfg.RANDOM_SEED,
    )
    contaminated_registry = all_images.copy()
    contaminated_registry.iloc[contaminated_train_idx, contaminated_registry.columns.get_loc("split")] = "train"
    contaminated_registry.iloc[contaminated_test_idx,  contaminated_registry.columns.get_loc("split")] = "test"
    # No val in contaminated condition — use small portion of train as val proxy
    n_val = max(1, len(contaminated_train_idx) // 10)
    val_from_train = contaminated_train_idx[:n_val]
    contaminated_registry.iloc[val_from_train, contaminated_registry.columns.get_loc("split")] = "val"

    loaders_c, _ = get_dataloaders(contaminated_registry, cfg)
    cw_c         = compute_class_weights(contaminated_registry, cfg.NUM_CLASSES)
    model_c      = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
    train(cfg, model_c, loaders_c, cw_c)
    metrics_c = evaluate(
        model=model_c,
        dataloader=loaders_c["test"],
        class_names=cfg.CLASS_NAMES,
        device=DEVICE,
        split_name="test_e2_contaminated",
        backbone=cfg.BACKBONE,
        metrics_dir=cfg.METRICS_DIR,
        figures_dir=cfg.FIGURES_DIR,
    )
    results["contaminated"] = {
        "accuracy":  metrics_c["accuracy"],
        "macro_f1":  metrics_c["macro_f1"],
        "note":      "INFLATED — augmented images leak into test split",
    }

    # --- Condition B: Leakage-safe split (standard protocol) ---
    print("\n--- E2: Condition 'leakage_safe' (originals-only split) ---")

    safe_registry = registry[registry["split"].notna()].copy()
    loaders_s, _ = get_dataloaders(safe_registry, cfg)
    cw_s         = compute_class_weights(safe_registry, cfg.NUM_CLASSES)
    model_s      = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
    train(cfg, model_s, loaders_s, cw_s)
    metrics_s = evaluate(
        model=model_s,
        dataloader=loaders_s["test"],
        class_names=cfg.CLASS_NAMES,
        device=DEVICE,
        split_name="test_e2_safe",
        backbone=cfg.BACKBONE,
        metrics_dir=cfg.METRICS_DIR,
        figures_dir=cfg.FIGURES_DIR,
    )
    results["leakage_safe"] = {
        "accuracy":  metrics_s["accuracy"],
        "macro_f1":  metrics_s["macro_f1"],
        "note":      "CORRECT — split on originals only",
    }

    # Compute inflation
    acc_inflation = (
        results["contaminated"]["accuracy"] - results["leakage_safe"]["accuracy"]
    ) * 100
    f1_inflation  = (
        results["contaminated"]["macro_f1"] - results["leakage_safe"]["macro_f1"]
    ) * 100
    results["accuracy_inflation_pp"] = acc_inflation
    results["f1_inflation_pp"]       = f1_inflation

    # Print paper-ready ablation table  # PRD §13
    print(f"\n{'='*60}")
    print(f"  ABLATION E2 — LEAKAGE CONTAMINATION IMPACT")
    print(f"{'='*60}")
    print(f"Leakage-Contaminated Split:")
    print(f"  Accuracy:   {results['contaminated']['accuracy']*100:.2f}%   "
          f"Macro-F1: {results['contaminated']['macro_f1']*100:.2f}%")
    print(f"\nLeakage-Safe Split (CORRECT):")
    print(f"  Accuracy:   {results['leakage_safe']['accuracy']*100:.2f}%   "
          f"Macro-F1: {results['leakage_safe']['macro_f1']*100:.2f}%")
    print(f"\nAccuracy Inflation from Leakage: +{acc_inflation:.2f} pp")
    print(f"{'='*60}")
    print(f"⚠ PAPER CLAIM: Report this inflation figure as primary contribution")
    print(f"{'='*60}\n")

    _save_ablation_result(results, "e2_leakage", cfg.METRICS_DIR)
    return results


def experiment_e3_class_weighting(cfg, registry: pd.DataFrame) -> dict:
    """
    E3: Class-weighted loss vs. unweighted loss.
    Focus on Neem recall improvement.
    """
    print("\n" + "="*60)
    print("  ABLATION E3 — CLASS WEIGHTING IMPACT")
    print("="*60)
    
    # Check if already completed
    if _check_ablation_completed("e3_class_weighting", cfg.METRICS_DIR):
        return _load_ablation_result("e3_class_weighting", cfg.METRICS_DIR)

    safe_registry = registry[registry["split"].notna()].copy()
    loaders, datasets = get_dataloaders(safe_registry, cfg)
    results = {}

    for weighted in [True, False]:
        tag  = "weighted" if weighted else "unweighted"
        print(f"\n--- E3: {tag} loss ---")

        if weighted:
            cw = compute_class_weights(safe_registry, cfg.NUM_CLASSES)
        else:
            # Uniform weights (no class rebalancing)
            cw = torch.ones(cfg.NUM_CLASSES)
            print("[E3] Using uniform (unweighted) class weights.")

        model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
        train(cfg, model, loaders, cw)
        metrics = evaluate(
            model=model,
            dataloader=loaders["test"],
            class_names=cfg.CLASS_NAMES,
            device=DEVICE,
            split_name=f"test_e3_{tag}",
            backbone=cfg.BACKBONE,
            metrics_dir=cfg.METRICS_DIR,
            figures_dir=cfg.FIGURES_DIR,
        )
        results[tag] = {
            "macro_f1":  metrics["macro_f1"],
            "accuracy":  metrics["accuracy"],
            "neem_f1":   metrics["per_class_f1"].get("Neem Leaf", 0.0),
            "per_class_f1": metrics["per_class_f1"],
        }

    delta_neem = (results["weighted"]["neem_f1"] - results["unweighted"]["neem_f1"]) * 100
    results["neem_f1_improvement_pp"] = delta_neem

    print(f"\n[E3] Weighted Neem F1:   {results['weighted']['neem_f1']*100:.2f}%")
    print(f"[E3] Unweighted Neem F1: {results['unweighted']['neem_f1']*100:.2f}%")
    print(f"[E3] Neem F1 improvement from weighting: +{delta_neem:.2f} pp")

    _save_ablation_result(results, "e3_class_weighting", cfg.METRICS_DIR)
    return results


def experiment_e4_augmentation_types(cfg, registry: pd.DataFrame) -> dict:
    """
    E4: Individual augmentation type ablation.
    Tests each transform independently.
    Note: VFlip may degrade performance (unnatural for leaves).
    """
    print("\n" + "="*60)
    print("  ABLATION E4 — AUGMENTATION TYPE ABLATION")
    print("="*60)
    
    # Check if already completed
    if _check_ablation_completed("e4_augmentation_types", cfg.METRICS_DIR):
        return _load_ablation_result("e4_augmentation_types", cfg.METRICS_DIR)

    import torchvision.transforms as T
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter

    safe_registry = registry[registry["split"].notna()].copy()
    results = {}

    mean, std = cfg.IMAGENET_MEAN, cfg.IMAGENET_STD
    base = [T.Resize(cfg.INPUT_SIZE), T.CenterCrop(cfg.INPUT_SIZE)]
    tail = [T.ToTensor(), T.Normalize(mean=mean, std=std)]

    # Each condition: name → list of train transforms to add between base and tail
    conditions = {
        "hflip_only":     [T.RandomHorizontalFlip(p=1.0)],
        "vflip_only":     [T.RandomVerticalFlip(p=1.0)],
        "rotation_only":  [T.RandomRotation(degrees=15)],
        "colorjitter_only": [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
        "all_combined":   [
            T.RandomResizedCrop(cfg.INPUT_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=15),
        ],
        "no_augmentation": [],
    }

    val_test_transform = T.Compose(base + tail)

    for cond_name, aug_ops in conditions.items():
        print(f"\n--- E4: {cond_name} ---")
        train_transform = T.Compose(base + aug_ops + tail)

        train_ds = MedicinalLeafDataset(
            safe_registry[safe_registry["split"] == "train"].reset_index(drop=True),
            transform=train_transform,
        )
        val_ds = MedicinalLeafDataset(
            safe_registry[safe_registry["split"] == "val"].reset_index(drop=True),
            transform=val_test_transform,
        )
        test_ds = MedicinalLeafDataset(
            safe_registry[safe_registry["split"] == "test"].reset_index(drop=True),
            transform=val_test_transform,
        )

        train_labels  = safe_registry[safe_registry["split"] == "train"]["label_idx"].values
        label_counts  = Counter(train_labels)
        sample_weights = torch.DoubleTensor([1.0 / label_counts[l] for l in train_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        loaders = {
            "train": DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                                num_workers=4, pin_memory=True, drop_last=True),
            "val":   DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True),
            "test":  DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True),
        }

        cw    = compute_class_weights(safe_registry, cfg.NUM_CLASSES)
        model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
        train(cfg, model, loaders, cw)
        metrics = evaluate(
            model=model,
            dataloader=loaders["test"],
            class_names=cfg.CLASS_NAMES,
            device=DEVICE,
            split_name=f"test_e4_{cond_name}",
            backbone=cfg.BACKBONE,
            metrics_dir=cfg.METRICS_DIR,
            figures_dir=cfg.FIGURES_DIR,
        )
        results[cond_name] = {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        }

        if cond_name == "vflip_only":
            baseline_f1 = results.get("no_augmentation", {}).get("macro_f1", 0.0)
            if metrics["macro_f1"] < baseline_f1:
                print(f"  ⚠ NOTE: VFlip DEGRADES performance "
                      f"({metrics['macro_f1']*100:.2f}% < {baseline_f1*100:.2f}%). "
                      f"As expected for leaf images — report in paper.")

    print("\n--- E4 Summary ---")
    for cond, res in results.items():
        print(f"  {cond:<25} Macro-F1={res['macro_f1']*100:.2f}%  Acc={res['accuracy']*100:.2f}%")

    _save_ablation_result(results, "e4_augmentation_types", cfg.METRICS_DIR)
    return results


def experiment_e5_transfer_learning(cfg, registry: pd.DataFrame) -> dict:
    """
    E5: ImageNet pretrained vs. random initialization.
    Expected: pretrained significantly outperforms at N=1,160.
    """
    print("\n" + "="*60)
    print("  ABLATION E5 — TRANSFER LEARNING IMPACT")
    print("="*60)
    
    # Check if already completed
    if _check_ablation_completed("e5_transfer_learning", cfg.METRICS_DIR):
        return _load_ablation_result("e5_transfer_learning", cfg.METRICS_DIR)

    safe_registry = registry[registry["split"].notna()].copy()
    loaders, _    = get_dataloaders(safe_registry, cfg)
    cw            = compute_class_weights(safe_registry, cfg.NUM_CLASSES)

    results = {}

    for pretrained in [True, False]:
        tag   = "pretrained" if pretrained else "random_init"
        print(f"\n--- E5: {tag} ---")

        model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, pretrained=pretrained)
        t0 = time.time()
        train(cfg, model, loaders, cw)
        training_time = time.time() - t0

        metrics = evaluate(
            model=model,
            dataloader=loaders["test"],
            class_names=cfg.CLASS_NAMES,
            device=DEVICE,
            split_name=f"test_e5_{tag}",
            backbone=cfg.BACKBONE,
            metrics_dir=cfg.METRICS_DIR,
            figures_dir=cfg.FIGURES_DIR,
        )
        results[tag] = {
            "macro_f1":      metrics["macro_f1"],
            "accuracy":      metrics["accuracy"],
            "training_time": training_time,
        }

    delta = (results["pretrained"]["macro_f1"] - results["random_init"]["macro_f1"]) * 100
    results["pretrained_advantage_pp"] = delta

    print(f"\n[E5] Pretrained  macro_f1={results['pretrained']['macro_f1']*100:.2f}%  "
          f"time={results['pretrained']['training_time']:.0f}s")
    print(f"[E5] Random init macro_f1={results['random_init']['macro_f1']*100:.2f}%  "
          f"time={results['random_init']['training_time']:.0f}s")
    print(f"[E5] Pretrained advantage: +{delta:.2f} pp Macro-F1")
    if delta > 5.0:
        print("  ✓ Significant pretrained advantage — confirm report in paper.")

    _save_ablation_result(results, "e5_transfer_learning", cfg.METRICS_DIR)
    return results
