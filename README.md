# Medicinal Plant Leaf Dataset: [Medicinal Plant Leaves (Mendeley Data)](https://data.mendeley.com/datasets/fj93rrfv2y/1)

## Table of Contentsassification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive deep learning pipeline for classifying medicinal plant leaves using transfer learning and advance## Repository Contents

This repository includes:
-  **Source Code**: Complete implementation (`src/`, `main.py`, `config.py`)
-  **Documentation**: Comprehensive README with complete usage guide
-  **Results**: All metrics (JSON), figures (PNG), confusion matrices, ROC curves
-  **Analysis Scripts**: Cross-validation analysis, ablation test scripts
-  **Dataset**: Not included (download from [Mendeley Data](https://data.mendeley.com/datasets/fj93rrfv2y/1))
-  **Model Weights**: Not included due to size (can be regenerated via training)

---echniques. This system implements rigorous data leakage prevention protocols and achieves state-of-the-art performance on botanical classification tasks.

**Repository:** [github.com/nishitbohra/Medicinal-Plant-Classification-LeakageSafe](https://github.com/nishitbohra/Medicinal-Plant-Classification-LeakageSafe)

**Dataset:** [Medicinal Plant Leaves (Mendeley Data)](https://data.mendeley.com/datasets/fj93rrfv2y/1)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Results](#results)
- [Research Contributions](#research-contributions)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a state-of-the-art medicinal plant leaf classification system designed to identify six different plant species through their leaf images. The system addresses critical challenges in botanical classification including data leakage prevention, class imbalance, and robust evaluation metrics.

### Supported Plant Classes
- Arjun Leaf (Terminalia arjuna)
- Curry Leaf (Murraya koenigii)
- Marsh Pennywort Leaf (Centella asiatica)
- Mint Leaf (Mentha)
- Neem Leaf (Azadirachta indica)
- Rubble Leaf

## Features

### Core Capabilities
- **Multi-Model Architecture Support**: EfficientNet-B4, ResNet50, VGG16, MobileNet-V3, Vision Transformer (ViT), Swin Transformer
- **Advanced Data Preprocessing**: EXIF metadata stripping, perceptual hash deduplication, resolution auditing
- **Robust Data Splitting**: Stratified splitting on original images only to prevent data leakage
- **Class Imbalance Handling**: Weighted random sampling and class-weighted loss functions
- **Comprehensive Evaluation**: Macro-F1, weighted-F1, accuracy with confidence intervals
- **Visualization**: Grad-CAM heatmaps, confusion matrices, ROC curves, training curves

### Research Features
- **Data Leakage Prevention**: Strict separation of original and augmented images during splitting
- **Ablation Studies**: Five comprehensive experiments examining augmentation impact, leakage effects, and model components
- **Cross-Validation**: K-fold validation with multiple backbone architectures
- **Transfer Learning**: Progressive unfreezing strategy with warm-up epochs

## Dataset

### Source
This project uses the **Medicinal Plant Leaf Dataset** from:
```
Dataset Citation:
Hossain, Molla Shahadat; Al-Hammadi, Mohammed; Khaled, Alabdulkareem; Hasan, Md Shamim;
Guizani, Mohsen (2024), "Medicinal Plant Leaves", Mendeley Data, V1
DOI: 10.17632/fj93rrfv2y.1
URL: https://data.mendeley.com/datasets/fj93rrfv2y/1
```

**Note**: The dataset is not included in this repository due to size. Please download it from the above link and place it in `data/dataset/`.

### Structure
```
data/dataset/
├── Original Images/          # Source images (1,160 total)
│   ├── Arjun Leaf/          # 220 images
│   ├── Curry Leaf/          # 165 images
│   ├── Marsh Pennywort Leaf/# 210 images
│   ├── Mint Leaf/           # 220 images
│   ├── Neem Leaf/           # 70 images (class imbalance)
│   └── Rubble Leaf/         # 275 images
├── Augmented Images/        # Generated augmentations
└── Processed Images/        # Resized to 224x224
```

### Data Statistics
- **Total Images**: 8,790 (1,160 original + 7,630 augmented)
- **Resolution Range**: 4000x3000 to 4032x3024 pixels
- **Split Ratios**: 70% train / 15% validation / 15% test
- **Class Imbalance**: Neem Leaf significantly underrepresented (70 images vs 220-275 for others)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for training)
- 16GB+ RAM recommended for full dataset processing

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/nishitbohra/Medicinal-Plant-Classification-LeakageSafe.git
cd Medicinal-Plant-Classification-LeakageSafe

# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Mendeley Data
# Place in: data/dataset/
# URL: https://data.mendeley.com/datasets/fj93rrfv2y/1

# Verify installation
python main.py --help
```

### Manual Installation
```bash
# Core ML/Deep Learning
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.21.0 scikit-learn>=1.0.0

# Data Processing
pip install pandas>=1.3.0 Pillow>=8.3.0 scipy>=1.7.0

# Image Processing
pip install piexif>=1.1.3 imagehash>=4.2.0

# Visualization
pip install matplotlib>=3.5.0 seaborn>=0.11.0
```

## Usage

### Command Line Interface

The system provides a modular CLI for running different pipeline stages:

```bash
# Run complete pipeline
python main.py --stage all

# Run individual stages
python main.py --stage preprocess    # Data preprocessing only
python main.py --stage train         # Training only
python main.py --stage evaluate      # Evaluation only
python main.py --stage gradcam       # Generate Grad-CAM visualizations
python main.py --stage ablation      # Run ablation studies
python main.py --stage cv            # Cross-validation experiments

# Override model backbone
python main.py --stage train --backbone resnet50
python main.py --stage train --backbone mobilenet_v3_large

# Force rebuild preprocessing cache
python main.py --stage preprocess --rebuild
```

### Supported Backbones
- `efficientnet_b4` (Primary model)
- `resnet50` (CNN baseline)
- `vgg16` (Historical baseline)
- `mobilenet_v3_large` (Mobile/edge deployment)
- `vit_b_16` (Vision Transformer)
- `swin_t` (Hierarchical Transformer)

### Quick Start Example
```bash
# 1. Ensure dataset is in correct location
ls data/dataset/"Original Images"/

# 2. Run preprocessing and training
python main.py --stage all --backbone efficientnet_b4

# 3. View results
ls outputs/figures/     # Visualizations
ls outputs/metrics/     # Performance metrics
ls outputs/checkpoints/ # Saved models
```

## Project Structure

```
medicinal_plant_clf/
├── main.py                 # Entry point and pipeline orchestration
├── config.py               # Configuration parameters
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── src/                   # Source code modules
│   ├── preprocessing.py   # EXIF stripping, deduplication, resizing
│   ├── split.py          # Data splitting with leakage prevention
│   ├── dataset.py        # PyTorch Dataset and DataLoader
│   ├── models.py         # Model architectures and factory
│   ├── train.py          # Training loop with early stopping
│   ├── evaluate.py       # Comprehensive evaluation metrics
│   ├── gradcam.py        # Grad-CAM visualization
│   ├── ablation.py       # Ablation study experiments
│   └── cross_validate.py # K-fold cross-validation
│
├── data/                 # Dataset directory
│   └── dataset/
│       ├── Original Images/
│       ├── Augmented Images/
│       └── Processed Images/
│
└── outputs/              # Generated outputs
    ├── checkpoints/      # Saved model weights
    ├── figures/          # Plots and visualizations
    └── metrics/          # Performance metrics and logs
```

## Models

### Primary Architecture: EfficientNet-B4
- **Parameters**: 17.5M total (10.7K trainable initially)
- **Strategy**: Transfer learning with progressive unfreezing
- **Warm-up**: 5 epochs with frozen backbone
- **Fine-tuning**: Full model training from epoch 6

### Training Strategy
1. **Phase 1 (Epochs 1-5)**: Frozen backbone, train classification head only
2. **Phase 2 (Epochs 6+)**: Unfreeze all layers, reduced learning rate
3. **Early Stopping**: Monitor validation macro-F1 with patience=15

### Optimization Details
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing Warm Restarts
- **Loss Function**: Cross-entropy with class weights
- **Batch Size**: 32 (adjustable in config.py)
- **Learning Rate**: 1e-4 (0.1x after unfreezing)

## Pipeline Stages

### 1. Preprocessing
- **EXIF Stripping**: Remove metadata to prevent acquisition artifacts
- **Resolution Auditing**: Analyze and validate image dimensions
- **Deduplication**: Remove perceptual duplicates using pHash (threshold=10)
- **Resizing**: Standardize to 224x224 pixels
- **Registry Creation**: Build path-label mapping with augmentation flags

### 2. Data Splitting
- **Strategy**: Stratified splits on original images only
- **Augmentation Handling**: Augmented images assigned to same split as parent
- **Validation**: Ensures no data leakage between train/val/test
- **Output**: Registry with split assignments saved to CSV

### 3. Training
- **Data Loading**: WeightedRandomSampler for class balance
- **Monitoring**: Real-time validation metrics
- **Checkpointing**: Save best model based on macro-F1
- **Logging**: Training history saved as JSON

### 4. Evaluation
- **Metrics**: Accuracy, macro-F1, weighted-F1 with 95% confidence intervals
- **Visualizations**: Confusion matrix, ROC curves, per-class performance
- **Statistical Analysis**: Bootstrapped confidence intervals

### 5. Grad-CAM Visualization
- **Attention Maps**: Highlight important image regions
- **Multi-Class**: Generate heatmaps for all plant classes
- **Interpretability**: Understand model decision patterns

## Configuration

All hyperparameters are centralized in `config.py`:

### Key Parameters
```python
# Model Configuration
BACKBONE = "efficientnet_b4"
PRETRAINED = True
INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100

# Data Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Early Stopping
EARLY_STOP_PAT = 15
EARLY_STOP_MON = "val_macro_f1"

# Deduplication
PHASH_THRESHOLD = 10
```

### Customization
Modify `config.py` to adjust:
- Model architectures and hyperparameters
- Data split ratios and preprocessing settings
- Training parameters and early stopping criteria
- File paths and output directories

## Results

### Performance Metrics
Results are automatically saved to `outputs/metrics/` including:
- **Test Metrics**: Accuracy, F1-scores with confidence intervals
- **Training History**: Loss curves, learning rate schedule
- **Per-Class Analysis**: Individual class performance breakdown

### Generated Outputs
- **Confusion Matrix**: Visual classification performance
- **ROC Curves**: Multi-class receiver operating characteristics
- **Training Curves**: Loss and metric progression
- **Grad-CAM Heatmaps**: Model attention visualization
- **Resolution Histogram**: Dataset image size distribution

## Research Contributions

### Novel Methodological Approaches
1. **Data Leakage Prevention Protocol**: Systematic approach to prevent augmented image contamination
2. **Class Imbalance Handling**: Comprehensive strategy combining sampling and loss weighting
3. **Progressive Transfer Learning**: Optimized unfreezing strategy for botanical classification
4. **Evaluation Robustness**: Macro-F1 focus with confidence interval reporting

### Experimental Design
- **E1**: Augmentation value quantification
- **E2**: Data leakage contamination impact (Primary contribution)
- **E3**: Class weighting effectiveness
- **E4**: Augmentation technique comparison
- **E5**: Transfer learning architecture comparison

## Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **Torchvision**: Pre-trained models and transforms
- **Scikit-learn**: Metrics and evaluation tools
- **Pandas**: Data manipulation and CSV handling
- **PIL/Pillow**: Image processing
- **piexif**: EXIF metadata manipulation
- **imagehash**: Perceptual hashing for deduplication

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slow training)
- **Recommended**: 16GB+ RAM, CUDA-compatible GPU
- **Storage**: 10GB+ free space for datasets and outputs

### Performance Considerations
- **CPU Training**: Very slow for large models (EfficientNet-B4)
- **GPU Acceleration**: Significant speedup recommended
- **Memory Usage**: Scales with batch size and model complexity
- **Disk I/O**: Preprocessing creates temporary files

## Troubleshooting

### Common Issues

#### 1. Memory Errors During Training
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or 8

# Or use lighter model
python main.py --backbone mobilenet_v3_large
```

#### 2. Slow Training on CPU
```bash
# Use mobile-optimized model
python main.py --backbone mobilenet_v3_large

# Or train only classification head
# Modify config.py: EPOCHS = 5 (skip unfreezing)
```

#### 3. Missing Augmented Folders
```
Warning: Augmented class folder not found
```
Solution: Ensure augmented dataset exists or modify `AUGMENTED_FOLDER_MAP` in config.py

#### 4. CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE = 8

# Enable gradient checkpointing (add to models.py if needed)
```

### Debug Mode
```bash
# Enable detailed logging
python -u main.py --stage all 2>&1 | tee training.log

# Check GPU usage
nvidia-smi

# Monitor memory usage
htop
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Code formatting
black src/ *.py

# Linting
flake8 src/ *.py

# Run tests (if available)
pytest tests/
```

### Guidelines
1. Follow existing code style and documentation patterns
2. Add comprehensive docstrings to new functions
3. Update configuration in `config.py` for new parameters
4. Include appropriate error handling and logging
5. Test changes with multiple backbone architectures

### Extending the System
- **New Models**: Add architectures to `src/models.py`
- **New Metrics**: Extend evaluation in `src/evaluate.py`
- **New Augmentations**: Modify transforms in `src/dataset.py`
- **New Experiments**: Add studies to `src/ablation.py`

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.

### Citation
If you use this code or methodology in your research, please cite:
```bibtex
@software{medicinal_plant_classification_2024,
  author = {Bohra, Nishit},
  title = {Medicinal Plant Leaf Classification System with Data Leakage Prevention},
  year = {2024},
  url = {https://github.com/nishitbohra/Medicinal-Plant-Classification-LeakageSafe},
  note = {Research-grade implementation with comprehensive ablation studies}
}
```

### Dataset Citation
```bibtex
@data{mendeley_medicinal_plants_2024,
  author = {Hossain, Molla Shahadat and Al-Hammadi, Mohammed and Khaled, Alabdulkareem and Hasan, Md Shamim and Guizani, Mohsen},
  title = {Medicinal Plant Leaves},
  year = {2024},
  publisher = {Mendeley Data},
  version = {V1},
  doi = {10.17632/fj93rrfv2y.1},
  url = {https://data.mendeley.com/datasets/fj93rrfv2y/1}
}
```

## Repository Contents

This repository includes:
-  **Source Code**: Complete implementation (`src/`, `main.py`, `config.py`)
-  **Documentation**: Research paper, figures, CV summary, submission checklist
-  **Results**: All metrics (JSON), figures (PNG), confusion matrices, ROC curves
-  **Analysis Scripts**: Cross-validation analysis, ablation test scripts
-  **Dataset**: Not included (download from [Mendeley Data](https://data.mendeley.com/datasets/fj93rrfv2y/1))
-  **Model Weights**: Not included due to size (can be regenerated via training)

## Related Files
- `RESEARCH_PAPER_Q1_DRAFT.md` - Full Q1 journal paper with methodology and results
- `CV_SUMMARY.md` - Cross-validation experimental results
- `FIGURES.md` - Publication-ready Mermaid diagrams
- `paper_submission_checklist.md` - Journal submission guidelines
- `latex_template/main.tex` - LaTeX template for paper formatting
- `COVER_LETTER_TEMPLATE.md` - Cover letter for journal submission

---

**For questions, issues, or contributions**, please open an issue on GitHub or contact the development team.

**Maintained by**: Nishit Bohra  
**Last Updated**: 2024
