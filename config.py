# config.py — Single source of truth for all hyperparameters and paths
# DO NOT hardcode any value elsewhere — always import from here

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "dataset"
ORIGINAL_DIR = DATA_DIR / "Original Images"
AUGMENTED_DIR = DATA_DIR / "Augmented Images"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Class definitions — canonical names (corrects Neem Leag typo via folder remapping)
CLASS_NAMES = [
    "Arjun Leaf",
    "Curry Leaf",
    "Marsh Pennywort Leaf",
    "Mint Leaf",
    "Neem Leaf",       # Note: augmented folder is named "Neem Leag" — handled in preprocessing
    "Rubble Leaf",
]
NUM_CLASSES = 6
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Augmented folder name mapping (corrects typo without modifying disk files)
# PRD §2.1 — 🔴 CRITICAL: "Neem Leag" is a known typo in the augmented folder
AUGMENTED_FOLDER_MAP = {
    "Arjun Leaf":           "Arjun Leaf",
    "Curry Leaf":           "Curry Leaf",
    "Marsh Pennywort Leaf": "Marsh Pennywort Leaf",
    "Mint Leaf":            "Mint Leaf",
    "Neem Leaf":            "Neem Leag",   # 🔴 CRITICAL: Typo in folder name — remapped here
    "Rubble Leaf":          "Rubble Leaf",
}

# Split ratios (applied on ORIGINALS ONLY)  # PRD §4.1
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# Per-class original counts (for reference and validation)
CLASS_COUNTS = {
    "Arjun Leaf":           220,
    "Curry Leaf":           165,
    "Marsh Pennywort Leaf": 210,
    "Mint Leaf":            220,
    "Neem Leaf":            70,   # ⚠ Critically small — class imbalance risk
    "Rubble Leaf":          275,
}

# Model training
BACKBONE        = "efficientnet_b4"   # Primary model
PRETRAINED      = True
INPUT_SIZE      = (224, 224)
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
EPOCHS          = 100
EARLY_STOP_PAT  = 15
EARLY_STOP_MON  = "val_macro_f1"    # Monitor Macro-F1, NOT accuracy  # PRD §6.2

# Scheduler
SCHEDULER       = "CosineAnnealingWarmRestarts"
T_0             = 10

# ImageNet normalization stats (used for transfer learning)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# pHash deduplication threshold (Hamming distance)  # PRD §3.2
PHASH_THRESHOLD = 10

# Cross-validation
CV_FOLDS        = 5

# Baseline architectures for comparison  # PRD §7.1
BASELINES = ["vgg16", "resnet50", "mobilenet_v3_large", "vit_b_16", "swin_t"]
