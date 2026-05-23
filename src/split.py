# src/split.py — Leakage-safe stratified split on originals only
# pip install scikit-learn pandas

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def _infer_original_stem(aug_path: str) -> str:
    """
    Derives the stem of a parent original from an augmented filename.
    Augmented naming convention: '<ClassName> Aug (N).jpg'
    Original naming convention:  '<ClassName> (N).JPG'
    Returns the stem without extension, e.g. 'Arjun Leaf (3)'.
    """
    stem = Path(aug_path).stem   # e.g. "Arjun Leaf Aug (3)"
    # Remove ' Aug' to recover original stem
    original_stem = re.sub(r"\s+Aug", "", stem, flags=re.IGNORECASE)
    # 🔴 CRITICAL: augmented Neem files are named "Neem Leag …" (typo) but originals
    # are "Neem Leaf …" — normalize so parent-lookup succeeds
    original_stem = original_stem.replace("Neem Leag", "Neem Leaf")
    return original_stem.strip()


def stratified_split_on_originals(
    registry: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Performs leakage-safe stratified split.

    CRITICAL RULE: Split is performed ONLY on original images (is_augmented == False).
    Augmented images inherit the split of their parent original.
    Val and Test NEVER include augmented images.

    Algorithm:
    1. Filter registry to original images only.
    2. Stratified split with sklearn (stratify=label_idx).
    3. Assign 'train'/'val'/'test' to originals.
    4. Augmented images: inherit parent's split; ONLY train-split parents used.
    5. Return updated registry DataFrame.
    """  # PRD §4.1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    registry = registry.copy()
    registry["split"] = None

    # Step 1: Work only with originals  # PRD §4.1 — CRITICAL
    originals = registry[~registry["is_augmented"]].copy()
    total_originals = len(originals)

    # Step 2: Stratified train / (val+test) split
    val_test_ratio = val_ratio + test_ratio
    train_idx, valtest_idx = train_test_split(
        originals.index,
        test_size=val_test_ratio,
        stratify=originals["label_idx"],
        random_state=seed,
    )

    # Step 3: Further split val+test into val / test
    val_within_ratio = val_ratio / val_test_ratio
    valtest_labels = originals.loc[valtest_idx, "label_idx"]
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=1.0 - val_within_ratio,
        stratify=valtest_labels,
        random_state=seed,
    )

    # Assign splits to originals
    registry.loc[train_idx, "split"] = "train"
    registry.loc[val_idx,   "split"] = "val"
    registry.loc[test_idx,  "split"] = "test"

    # Validate totals
    assigned = len(train_idx) + len(val_idx) + len(test_idx)
    assert assigned == total_originals, \
        f"Split count mismatch: {assigned} != {total_originals}"

    # Step 4: Propagate split to augmented images
    # Build map: original_stem → split
    original_stem_to_split = {}
    for _, row in registry[~registry["is_augmented"]].iterrows():
        stem = Path(row["path"]).stem
        original_stem_to_split[stem] = row["split"]

    aug_mask = registry["is_augmented"]
    aug_indices = registry[aug_mask].index

    for idx in aug_indices:
        aug_path = registry.at[idx, "path"]
        orig_stem = _infer_original_stem(aug_path)
        parent_split = original_stem_to_split.get(orig_stem, None)

        if parent_split == "train":
            # Only train-split augmented images are included  # PRD §4.2
            registry.at[idx, "split"] = "train"
        else:
            # Val/test augmented images are EXCLUDED (set to None = dropped)
            registry.at[idx, "split"] = None

    # Drop augmented images not in train (val/test must be originals only)
    registry = registry[
        (registry["split"].notna()) |
        (~registry["is_augmented"])
    ].copy()
    # After filtering, remove original val/test that might have None split
    registry = registry[registry["split"].notna()].copy()

    # Step 5: Print per-class split counts for verification  # PRD §5.3
    print("\n--- Per-Class Split Counts (originals only for val/test) ---")
    print(f"{'Class':<30} {'Train':>7} {'Val':>5} {'Test':>5} {'Total':>7}")
    print("-" * 58)

    orig_only = registry[~registry["is_augmented"]]
    for cls in sorted(registry["class_name"].unique()):
        cls_orig = orig_only[orig_only["class_name"] == cls]
        n_train = (cls_orig["split"] == "train").sum()
        n_val   = (cls_orig["split"] == "val").sum()
        n_test  = (cls_orig["split"] == "test").sum()
        total   = n_train + n_val + n_test

        # Raise if any class has 0 test images
        if n_test == 0:
            raise ValueError(
                f"Class '{cls}' has 0 test images. Check split ratios and class count."
            )

        # Warn loudly if Neem has < 15 test images  # PRD §4.3
        if n_test < 15:
            print(f"  ⚠ WARNING: '{cls}' has only {n_test} test images — "
                  f"CI will be very wide (±~31%). Report with caution.")

        print(f"  {cls:<30} {n_train:>7} {n_val:>5} {n_test:>5} {total:>7}")

    print("-" * 58)
    all_orig = orig_only
    total_train = (all_orig["split"] == "train").sum()
    total_val   = (all_orig["split"] == "val").sum()
    total_test  = (all_orig["split"] == "test").sum()
    print(f"  {'TOTAL (originals)':<30} {total_train:>7} {total_val:>5} {total_test:>5} "
          f"{total_train + total_val + total_test:>7}")

    aug_train = (registry[registry["is_augmented"]]["split"] == "train").sum()
    print(f"  Augmented images in train: {aug_train}")
    print("------------------------------------------------------------\n")

    return registry


def load_or_build_registry(cfg, rebuild: bool = False) -> pd.DataFrame:
    """
    Loads registry from CSV if it exists, otherwise rebuilds via preprocessing.
    Always re-runs split to ensure reproducibility.
    """
    registry_path = cfg.METRICS_DIR / "path_label_registry.csv"

    if registry_path.exists() and not rebuild:
        print(f"[Split] Loading existing registry from {registry_path}")
        registry = pd.read_csv(registry_path)
        registry["is_augmented"] = registry["is_augmented"].astype(bool)
    else:
        from src.preprocessing import run_preprocessing
        registry = run_preprocessing(cfg)

    # Always re-apply the split for consistency
    registry = stratified_split_on_originals(
        registry=registry,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        seed=cfg.RANDOM_SEED,
    )

    # Save updated registry with split assignments
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.METRICS_DIR / "registry_with_splits.csv"
    registry.to_csv(out_path, index=False)
    print(f"[Split] Registry with splits saved to {out_path}")

    return registry
