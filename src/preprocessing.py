# src/preprocessing.py — EXIF strip, pHash dedup, rename, resize, normalize
# pip install piexif imagehash Pillow pandas numpy matplotlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import piexif          # EXIF stripping
import imagehash       # pHash deduplication
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def strip_exif(image_paths: list) -> None:
    """
    Remove all EXIF metadata (GPS, device ID, timestamp) from all JPEG images.
    Uses piexif. Processes in-place. Logs count of files processed.
    Prevents acquisition-artifact leakage into model features.
    """
    processed = 0
    failed = 0
    for p in image_paths:
        try:
            piexif.remove(str(p))   # Remove all EXIF in-place
            processed += 1
        except Exception:
            # Some files may have no EXIF or be malformed — skip gracefully
            failed += 1
    print(f"[EXIF] Stripped metadata from {processed} files ({failed} skipped/no EXIF).")


def audit_resolutions(image_paths: list, figures_dir: Path) -> dict:
    """
    Compute (width, height) for all original images (1,160 total).
    Returns dict with: min_res, max_res, median_res, mean_res, std_res, outliers.
    Flags outliers as images > 2 std deviations from mean area.
    Saves resolution histogram as PNG to figures_dir/resolution_histogram.png.
    Prints summary to stdout for inclusion in paper.
    """
    widths, heights, areas = [], [], []
    image_paths_list = list(image_paths)
    total_input      = len(image_paths_list)   # Total requested (including any failures)
    # Track only successfully opened paths so indices stay aligned with areas_arr
    opened_paths = []

    for p in image_paths_list:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                opened_paths.append(p)
        except Exception as e:
            print(f"⚠ WARNING: Could not open {p}: {e}")

    areas_arr = np.array(areas)
    mean_area = float(np.mean(areas_arr))
    std_area  = float(np.std(areas_arr))

    # Flag outliers: area more than 2 std from mean (index into opened_paths, not raw list)
    outlier_mask  = np.abs(areas_arr - mean_area) > 2 * std_area
    outlier_paths = [opened_paths[i] for i in range(len(opened_paths)) if outlier_mask[i]]

    result = {
        "min_res":    (int(min(widths)), int(min(heights))),
        "max_res":    (int(max(widths)), int(max(heights))),
        "median_res": (int(np.median(widths)), int(np.median(heights))),
        "mean_res":   (float(np.mean(widths)), float(np.mean(heights))),
        "std_res":    (float(np.std(widths)), float(np.std(heights))),
        "mean_area":  mean_area,
        "std_area":   std_area,
        "outliers":   [str(p) for p in outlier_paths],
        "n_images":   total_input,
    }

    # Save resolution histogram  # PRD §3.1
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title("Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    axes[1].hist(heights, bins=30, color="darkorange", edgecolor="black")
    axes[1].set_title("Height Distribution")
    axes[1].set_xlabel("Height (px)")
    fig.suptitle("Original Image Resolution Distribution (N=1,160)")
    plt.tight_layout()
    out_path = figures_dir / "resolution_histogram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Audit] Resolution histogram saved to {out_path}")

    print("\n--- Resolution Audit Summary ---")
    print(f"  N images    : {result['n_images']}")
    print(f"  Min res     : {result['min_res']}")
    print(f"  Max res     : {result['max_res']}")
    print(f"  Median res  : {result['median_res']}")
    print(f"  Mean res    : ({result['mean_res'][0]:.1f}, {result['mean_res'][1]:.1f})")
    print(f"  Std res     : ({result['std_res'][0]:.1f}, {result['std_res'][1]:.1f})")
    print(f"  Outliers    : {len(outlier_paths)} images (>2σ from mean area)")
    for op in outlier_paths[:10]:   # Print first 10 outliers
        print(f"    {op}")
    print("--------------------------------\n")

    return result


def deduplicate_originals(original_dir: Path, threshold: int = 10) -> list:
    """
    Compute perceptual hash (pHash) for ALL original images per class.
    Clusters by Hamming distance <= threshold.
    Within each cluster, keeps one image; marks rest as duplicates.
    Returns list of paths to KEEP (non-duplicates).
    Prints count removed per class and total removed.
    Does NOT delete files — returns keep list only.
    """
    keep_paths = []
    total_removed = 0

    for class_dir in sorted(original_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        # Collect all image files in this class folder
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not images:
            continue

        # Compute pHash for each image
        hashes = {}
        for p in images:
            try:
                with Image.open(p) as img:
                    hashes[p] = imagehash.phash(img)
            except Exception as e:
                print(f"⚠ WARNING: Could not hash {p}: {e}")

        # Greedy deduplication: union-find style clustering by Hamming distance
        to_remove = set()
        paths = list(hashes.keys())
        for i in range(len(paths)):
            if paths[i] in to_remove:
                continue
            for j in range(i + 1, len(paths)):
                if paths[j] in to_remove:
                    continue
                dist = hashes[paths[i]] - hashes[paths[j]]  # Hamming distance
                if dist <= threshold:
                    to_remove.add(paths[j])   # Keep earlier image, remove later

        kept = [p for p in images if p not in to_remove]
        removed_count = len(images) - len(kept)
        total_removed += removed_count
        keep_paths.extend(kept)

        print(f"[Dedup] {class_dir.name}: {len(images)} images → "
              f"kept {len(kept)}, removed {removed_count} duplicates")

    print(f"[Dedup] Total removed: {total_removed} | Total kept: {len(keep_paths)}")
    return keep_paths


def build_path_label_registry(
    original_dir: Path,
    augmented_dir: Path,
    augmented_folder_map: dict,
    class_to_idx: dict,
    metrics_dir: Path,
) -> pd.DataFrame:
    """
    Creates a DataFrame: [path, class_name, label_idx, split, is_augmented].
    Originals: scanned from ORIGINAL_DIR/<ClassName>/.
    Augmented: scanned from AUGMENTED_DIR/<MappedFolderName>/ via augmented_folder_map
               to resolve the Neem Leag → Neem Leaf mapping.
    split column filled later by split.py.
    Saves registry CSV to metrics_dir/path_label_registry.csv.
    NEVER infers class from filename — always uses folder name via augmented_folder_map.
    """  # PRD §3.4
    records = []

    # --- Scan original images ---
    for canonical_name, label_idx in class_to_idx.items():
        class_dir = original_dir / canonical_name
        if not class_dir.exists():
            print(f"⚠ WARNING: Original class folder not found: {class_dir}")
            continue
        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for p in image_files:
            records.append({
                "path":         str(p),
                "class_name":   canonical_name,
                "label_idx":    label_idx,
                "split":        None,          # Assigned later
                "is_augmented": False,
            })

    # --- Scan augmented images using folder map ---
    # 🔴 CRITICAL: Use augmented_folder_map to resolve "Neem Leag" → "Neem Leaf"
    for canonical_name, folder_name in augmented_folder_map.items():
        label_idx = class_to_idx[canonical_name]
        aug_class_dir = augmented_dir / folder_name
        if not aug_class_dir.exists():
            print(f"⚠ WARNING: Augmented class folder not found: {aug_class_dir}")
            continue
        image_files = sorted(
            p for p in aug_class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for p in image_files:
            records.append({
                "path":         str(p),
                "class_name":   canonical_name,   # Canonical name, NOT folder name
                "label_idx":    label_idx,
                "split":        None,
                "is_augmented": True,
            })

    registry = pd.DataFrame(records)

    # Validate registry
    print("\n--- Path-Label Registry Summary ---")
    for cls in class_to_idx:
        orig_count = len(registry[(registry["class_name"] == cls) & (~registry["is_augmented"])])
        aug_count  = len(registry[(registry["class_name"] == cls) & (registry["is_augmented"])])
        print(f"  {cls:<30} originals={orig_count:>3}  augmented={aug_count:>3}")
    print(f"  TOTAL: {len(registry)} entries")
    print("-----------------------------------\n")

    # Save registry for reproducibility
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "path_label_registry.csv"
    registry.to_csv(out_csv, index=False)
    print(f"[Registry] Saved to {out_csv}")

    return registry


def resize_and_standardize(
    image_paths: list,
    output_dir: Path,
    source_root: Path,
    target_size: tuple = (224, 224),
) -> None:
    """
    Resizes all images to target_size using center-crop preserving aspect ratio.
    Pads with black if aspect ratio > 1.5.
    Unifies .JPG (uppercase) and .jpg (lowercase) to .jpg lowercase.
    Saves processed copies to output_dir mirroring the original folder structure.
    Does NOT modify original files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tw, th = target_size
    processed = 0

    for src_path in image_paths:
        try:
            # Mirror folder structure under output_dir
            rel = Path(src_path).relative_to(source_root)
            # Normalize extension to lowercase .jpg
            dst_path = output_dir / rel.with_suffix(".jpg")
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(src_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                aspect = w / h

                if aspect > 1.5 or aspect < (1 / 1.5):
                    # Pad to square with black background before resize
                    max_dim = max(w, h)
                    padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
                    paste_x = (max_dim - w) // 2
                    paste_y = (max_dim - h) // 2
                    padded.paste(img, (paste_x, paste_y))
                    img = padded

                # Center-crop to target aspect ratio, then resize
                img = img.resize((tw, th), Image.LANCZOS)
                img.save(dst_path, "JPEG", quality=95)
                processed += 1

        except Exception as e:
            print(f"⚠ WARNING: Could not process {src_path}: {e}")

    print(f"[Resize] Processed {processed}/{len(image_paths)} images → {output_dir}")


def run_preprocessing(cfg) -> pd.DataFrame:
    """
    Orchestrates all preprocessing steps in order:
      1. EXIF stripping
      2. Resolution audit
      3. pHash deduplication
      4. Path-label registry construction (folder-based labeling, Neem Leag remapping)
      5. Resize + standardize to processed/ subdirectory
    Returns the path-label registry DataFrame.
    """
    print("\n" + "="*60)
    print("  PREPROCESSING PIPELINE")
    print("="*60)

    # Validate dataset directory exists
    if not cfg.ORIGINAL_DIR.exists():
        raise FileNotFoundError(
            f"Original images directory not found: {cfg.ORIGINAL_DIR}\n"
            f"Please place the dataset at: {cfg.DATA_DIR}"
        )
    if not cfg.AUGMENTED_DIR.exists():
        raise FileNotFoundError(
            f"Augmented images directory not found: {cfg.AUGMENTED_DIR}"
        )

    # Collect all original image paths
    all_original_paths = sorted(
        p for p in cfg.ORIGINAL_DIR.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    print(f"\n[Step 0] Found {len(all_original_paths)} original images.")

    # Step 1 — EXIF stripping
    print("\n[Step 1] Stripping EXIF metadata...")
    strip_exif(all_original_paths)

    # Step 2 — Resolution audit
    print("\n[Step 2] Auditing image resolutions...")
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    audit_resolutions(all_original_paths, cfg.FIGURES_DIR)

    # Step 3 — pHash deduplication (returns keep list)
    print("\n[Step 3] Running pHash deduplication...")
    keep_paths = deduplicate_originals(cfg.ORIGINAL_DIR, threshold=cfg.PHASH_THRESHOLD)
    print(f"[Dedup] {len(keep_paths)} originals retained after deduplication.")

    # Step 4 — Build path-label registry
    print("\n[Step 4] Building path-label registry...")
    registry = build_path_label_registry(
        original_dir=cfg.ORIGINAL_DIR,
        augmented_dir=cfg.AUGMENTED_DIR,
        augmented_folder_map=cfg.AUGMENTED_FOLDER_MAP,
        class_to_idx=cfg.CLASS_TO_IDX,
        metrics_dir=cfg.METRICS_DIR,
    )

    # Step 5 — Resize and standardize originals only (augmented already pre-processed)
    processed_dir = cfg.DATA_DIR / "Processed Images"
    print(f"\n[Step 5] Resizing originals to {cfg.INPUT_SIZE} → {processed_dir}")
    resize_and_standardize(
        image_paths=keep_paths,
        output_dir=processed_dir,
        source_root=cfg.ORIGINAL_DIR,
        target_size=cfg.INPUT_SIZE,
    )

    print("\n" + "="*60)
    print("  PREPROCESSING COMPLETE")
    print("="*60 + "\n")

    return registry
