# src/dataset.py — PyTorch Dataset class + DataLoader factory
# pip install torch torchvision pandas Pillow

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter


class MedicinalLeafDataset(Dataset):
    """
    PyTorch Dataset for medicinal leaf classification.

    Args:
        registry (pd.DataFrame): path_label_registry filtered to the desired split.
        transform: torchvision transforms to apply.

    Loads images from registry['path'] using PIL (RGB).
    Ground truth from registry['label_idx'].
    Handles missing files with a descriptive error.
    """

    def __init__(self, registry: pd.DataFrame, transform=None):
        # Clean the registry and ensure no NaN values
        self.registry = registry.reset_index(drop=True).copy()
        
        # Check for and handle NaN values in critical columns
        if self.registry["label_idx"].isna().any():
            print(f"[Dataset] Warning: Found NaN values in label_idx, dropping {self.registry['label_idx'].isna().sum()} rows")
            self.registry = self.registry.dropna(subset=["label_idx"]).reset_index(drop=True)
        
        # Ensure label_idx is integer
        self.registry["label_idx"] = self.registry["label_idx"].astype(int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.registry)

    def __getitem__(self, idx: int):
        row   = self.registry.iloc[idx]
        fpath = Path(row["path"])
        
        # Additional safety check for label_idx
        try:
            label = int(row["label_idx"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid label_idx at index {idx}: {row['label_idx']} "
                f"(type: {type(row['label_idx'])}). Path: {fpath}"
            )

        if not fpath.exists():
            raise FileNotFoundError(
                f"Image not found: {fpath}\n"
                f"Check that the dataset is correctly placed under data/dataset/"
            )

        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image {fpath}: {e}")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(split: str, input_size: tuple, mean: list, std: list):
    """
    Returns torchvision.transforms.Compose for the given split.

    Train: random augmentation (RandomResizedCrop, HFlip, ColorJitter, Rotation).
    Val/Test: deterministic Resize + CenterCrop only — NO random ops.
    """
    assert split in ("train", "val", "test"), f"Unknown split: {split}"

    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        # Val and Test must be IDENTICAL and deterministic  # PRD §5.1
        return T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def compute_class_weights(registry: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for CrossEntropyLoss.
    weight[c] = total_samples / (num_classes * count[c])
    Returns torch.FloatTensor of shape (num_classes,).
    Prints weights for all 6 classes.
    """  # PRD §6.1
    train_reg = registry[registry["split"] == "train"]
    counts    = train_reg["label_idx"].value_counts().sort_index()
    total     = len(train_reg)

    weights = []
    print("\n--- Class Weights (for CrossEntropyLoss) ---")
    for c in range(num_classes):
        count    = counts.get(c, 1)   # Avoid division by zero
        w        = total / (num_classes * count)
        weights.append(w)
        print(f"  Class {c}: count={count:>4}  weight={w:.4f}")
    print("--------------------------------------------\n")

    return torch.FloatTensor(weights)


def get_dataloaders(registry: pd.DataFrame, cfg) -> dict:
    """
    Builds and returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.

    Training DataLoader uses WeightedRandomSampler for stratified batch sampling.
    Val/Test DataLoaders are deterministic (shuffle=False).
    """
    transforms = {
        split: get_transforms(split, cfg.INPUT_SIZE, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
        for split in ("train", "val", "test")
    }

    datasets = {
        split: MedicinalLeafDataset(
            registry[registry["split"] == split].reset_index(drop=True),
            transform=transforms[split],
        )
        for split in ("train", "val", "test")
    }

    # WeightedRandomSampler for train  # PRD §6.1
    train_labels = registry[registry["split"] == "train"]["label_idx"].values
    label_counts  = Counter(train_labels)
    # sample_weights[i] = 1 / class_count[label[i]]
    sample_weights = torch.DoubleTensor(
        [1.0 / label_counts[lbl] for lbl in train_labels]
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        ),
    }

    # Verify WeightedRandomSampler class distribution (first 5 batches)
    print("\n[DataLoader] Verifying WeightedRandomSampler — first 5 train batches:")
    label_names = {v: k for k, v in cfg.CLASS_TO_IDX.items()}
    for i, (_, batch_labels) in enumerate(loaders["train"]):
        if i >= 5:
            break
        counts = Counter(batch_labels.tolist())
        dist   = {label_names.get(k, k): v for k, v in sorted(counts.items())}
        print(f"  Batch {i+1}: {dist}")
    print()

    for split, ds in datasets.items():
        print(f"[Dataset] {split}: {len(ds)} samples")

    return loaders, datasets
