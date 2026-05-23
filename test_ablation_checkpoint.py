"""
Test checkpoint functionality for ablation experiments
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg

from src.ablation import (
    _check_ablation_completed,
    _load_ablation_result
)

print("=" * 60)
print("Testing Ablation Checkpoint Functionality")
print("=" * 60)

experiments = [
    ("e1_augmentation", "E1: Augmentation Value"),
    ("e2_leakage", "E2: Data Leakage Impact"),
    ("e3_class_weighting", "E3: Class Weighting"),
    ("e4_augmentation_types", "E4: Augmentation Types"),
    ("e5_transfer_learning", "E5: Transfer Learning")
]

print("\nChecking which experiments are already completed:\n")

for exp_id, exp_name in experiments:
    completed = _check_ablation_completed(exp_id, cfg.METRICS_DIR)
    status = "✅ COMPLETED" if completed else "❌ PENDING"
    print(f"{status} - {exp_name}")
    
    if completed:
        result = _load_ablation_result(exp_id, cfg.METRICS_DIR)
        if result:
            print(f"   └─ Result preview: {list(result.keys())[:3]}...")

print("\n" + "=" * 60)
print("Summary:")
completed_count = sum(1 for exp_id, _ in experiments if _check_ablation_completed(exp_id, cfg.METRICS_DIR))
print(f"Completed: {completed_count}/5 experiments")
print(f"Remaining: {5 - completed_count}/5 experiments")
print("=" * 60)

if completed_count > 0:
    print(f"\n✓ When you run 'python main.py --stage ablation',")
    print(f"  it will skip {completed_count} completed experiment(s) and only run the remaining ones!")
else:
    print("\nNo experiments completed yet. All will run from scratch.")
