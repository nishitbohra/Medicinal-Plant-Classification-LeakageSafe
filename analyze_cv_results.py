"""
CV Results Analysis Script
Analyzes and visualizes cross-validation results from the medicinal plant classification pipeline.

Usage:
    python analyze_cv_results.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Setup
METRICS_DIR = Path("outputs/outputs/metrics")
FIGURES_DIR = Path("outputs/outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load fold results
fold_files = sorted(METRICS_DIR.glob("cv_fold*_metrics_efficientnet_b4_*.json"))
print(f"Found {len(fold_files)} fold result files")

fold_results = []
for ffile in fold_files:
    with open(ffile, 'r') as f:
        data = json.load(f)
        fold_results.append(data)
    print(f"Loaded: {ffile.name}")

if not fold_results:
    print("No CV fold results found!")
    exit(1)

# Extract metrics
folds = []
accuracies = []
macro_f1s = []
kappas = []
per_class_f1_data = {cls: [] for cls in fold_results[0]['per_class_f1'].keys()}

for i, fr in enumerate(fold_results, 1):
    folds.append(f"Fold {i}")
    accuracies.append(fr['accuracy'] * 100)
    macro_f1s.append(fr['macro_f1'] * 100)
    kappas.append(fr['cohen_kappa'])
    
    for cls, f1 in fr['per_class_f1'].items():
        per_class_f1_data[cls].append(f1 * 100)

# Print summary statistics
print("\n" + "="*70)
print("  CROSS-VALIDATION SUMMARY")
print("="*70)
print(f"\nNumber of completed folds: {len(fold_results)}")
print(f"\nOverall Metrics (Mean ± Std):")
print(f"  Accuracy:  {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
print(f"  Macro-F1:  {np.mean(macro_f1s):.2f}% ± {np.std(macro_f1s):.2f}%")
print(f"  Kappa:     {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")

print(f"\nPer-Class F1 Scores (Mean ± Std):")
for cls, scores in per_class_f1_data.items():
    print(f"  {cls:<30} {np.mean(scores):.2f}% ± {np.std(scores):.2f}%")

print("\n" + "="*70)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Overall metrics comparison
ax1 = axes[0, 0]
x = np.arange(len(folds))
width = 0.25
ax1.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
ax1.bar(x, macro_f1s, width, label='Macro-F1', alpha=0.8)
ax1.bar(x + width, [k*100 for k in kappas], width, label='Kappa (×100)', alpha=0.8)
ax1.set_xlabel('Fold')
ax1.set_ylabel('Score (%)')
ax1.set_title('Overall Metrics by Fold')
ax1.set_xticks(x)
ax1.set_xticklabels(folds)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([90, 102])

# 2. Per-class F1 scores
ax2 = axes[0, 1]
per_class_df = pd.DataFrame(per_class_f1_data, index=folds)
per_class_df.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_xlabel('Fold')
ax2.set_ylabel('F1 Score (%)')
ax2.set_title('Per-Class F1 Scores by Fold')
ax2.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([85, 105])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

# 3. Metric stability (coefficient of variation)
ax3 = axes[1, 0]
cv_data = {
    'Accuracy': np.std(accuracies) / np.mean(accuracies) * 100,
    'Macro-F1': np.std(macro_f1s) / np.mean(macro_f1s) * 100,
    'Kappa': np.std(kappas) / np.mean(kappas) * 100,
}
bars = ax3.bar(cv_data.keys(), cv_data.values(), alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax3.set_ylabel('Coefficient of Variation (%)')
ax3.set_title('Metric Stability Across Folds\n(Lower = More Stable)')
ax3.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom')

# 4. Per-class variability
ax4 = axes[1, 1]
class_means = [np.mean(scores) for scores in per_class_f1_data.values()]
class_stds = [np.std(scores) for scores in per_class_f1_data.values()]
class_names_short = [name.split()[0] for name in per_class_f1_data.keys()]  # Shorter names
x_pos = np.arange(len(class_names_short))
bars = ax4.bar(x_pos, class_means, yerr=class_stds, alpha=0.7, capsize=5, color='steelblue')
ax4.set_xlabel('Plant Class')
ax4.set_ylabel('F1 Score (%)')
ax4.set_title('Per-Class F1: Mean with Standard Deviation')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(class_names_short, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([85, 105])

plt.tight_layout()
output_path = FIGURES_DIR / "cv_analysis_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")
plt.close()

# Create detailed comparison table
comparison_df = pd.DataFrame({
    'Fold': folds,
    'Accuracy (%)': [f"{a:.2f}" for a in accuracies],
    'Macro-F1 (%)': [f"{m:.2f}" for m in macro_f1s],
    'Kappa': [f"{k:.4f}" for k in kappas],
})

print("\n" + "="*70)
print("  FOLD-BY-FOLD COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))
print("="*70)

# Save to CSV
csv_path = METRICS_DIR / "cv_comparison_summary.csv"
comparison_df.to_csv(csv_path, index=False)
print(f"\n✓ Comparison table saved to: {csv_path}")

# Identify best and worst performing folds
best_fold_idx = np.argmax(macro_f1s)
worst_fold_idx = np.argmin(macro_f1s)

print(f"\n📊 Key Insights:")
print(f"  Best Fold:  {folds[best_fold_idx]} (Macro-F1: {macro_f1s[best_fold_idx]:.2f}%)")
print(f"  Worst Fold: {folds[worst_fold_idx]} (Macro-F1: {macro_f1s[worst_fold_idx]:.2f}%)")
print(f"  Performance Range: {max(macro_f1s) - min(macro_f1s):.2f}%")

# Identify most/least stable classes
class_cvs = {cls: (np.std(scores) / np.mean(scores) * 100) 
             for cls, scores in per_class_f1_data.items()}
most_stable = min(class_cvs, key=class_cvs.get)
least_stable = max(class_cvs, key=class_cvs.get)

print(f"\n🎯 Per-Class Stability:")
print(f"  Most Stable:  {most_stable} (CV: {class_cvs[most_stable]:.2f}%)")
print(f"  Least Stable: {least_stable} (CV: {class_cvs[least_stable]:.2f}%)")

print("\n" + "="*70)
print("Analysis complete! Check the outputs directory for visualizations.")
print("="*70 + "\n")
