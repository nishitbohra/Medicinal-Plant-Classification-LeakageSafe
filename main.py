"""
main.py — Full pipeline entry point for medicinal plant leaf classification.

Usage:
    python main.py --stage all           # Run entire pipeline
    python main.py --stage preprocess    # Only preprocessing
    python main.py --stage train         # Only training
    python main.py --stage evaluate      # Only evaluation
    python main.py --stage ablation      # Only ablation experiments
    python main.py --stage cv            # Only cross-validation
    python main.py --stage gradcam       # Only Grad-CAM visualization
    python main.py --backbone resnet50   # Override backbone from config
    python main.py --resume              # Resume from checkpoint

pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn
pip install piexif imagehash Pillow scipy
"""

import argparse
import random
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# ─────────────────────────────────────────────────────────────────────────────

import config as cfg_module

from src.preprocessing  import run_preprocessing
from src.split          import stratified_split_on_originals
from src.dataset        import get_dataloaders, compute_class_weights
from src.models         import get_model
from src.train          import train
from src.evaluate       import evaluate
from src.gradcam        import visualize_gradcam_grid
from src.ablation       import (
    experiment_e1_augmentation_value,
    experiment_e2_leakage_contamination,
    experiment_e3_class_weighting,
    experiment_e4_augmentation_types,
    experiment_e5_transfer_learning,
)
from src.cross_validate import run_kfold_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(cfg, stage, backbone):
    """Setup comprehensive logging with file and console output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = cfg.METRICS_DIR / f"pipeline_log_{stage}_{backbone}_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('MedicinalPlantClassifier')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def save_pipeline_state(cfg, stage, backbone, state_data):
    """Save current pipeline state for resumability."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = cfg.METRICS_DIR / f"pipeline_state_{backbone}_{timestamp}.json"
    
    state = {
        "timestamp": timestamp,
        "stage": stage,
        "backbone": backbone,
        "device": str(DEVICE),
        "completed_stages": state_data.get("completed_stages", []),
        "current_epoch": state_data.get("current_epoch", 0),
        "best_metrics": state_data.get("best_metrics", {}),
        "training_history": state_data.get("training_history", {}),
        "checkpoint_path": state_data.get("checkpoint_path", ""),
        "registry_path": str(cfg.METRICS_DIR / "registry_with_splits.csv")
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    return state_file


def load_pipeline_state(cfg, backbone):
    """Load the most recent pipeline state for resuming."""
    state_pattern = f"pipeline_state_{backbone}_*.json"
    state_files = list(cfg.METRICS_DIR.glob(state_pattern))
    
    if not state_files:
        return None
    
    # Get most recent state file
    latest_state = max(state_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_state, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"Warning: Could not load pipeline state from {latest_state}: {e}")
        return None


def save_model_results(cfg, backbone, results, stage="evaluation"):
    """Save comprehensive model results in multiple formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON format - detailed results
    json_file = cfg.METRICS_DIR / f"results_{backbone}_{stage}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # CSV format - metrics table
    if "test_metrics" in results:
        metrics_df = pd.DataFrame([results["test_metrics"]])
        metrics_df["backbone"] = backbone
        metrics_df["timestamp"] = timestamp
        metrics_df["stage"] = stage
        
        csv_file = cfg.METRICS_DIR / f"metrics_{backbone}_{stage}_{timestamp}.csv"
        metrics_df.to_csv(csv_file, index=False)
        
        # Append to master results file
        master_csv = cfg.METRICS_DIR / "all_model_results.csv"
        if master_csv.exists():
            master_df = pd.read_csv(master_csv)
            master_df = pd.concat([master_df, metrics_df], ignore_index=True)
        else:
            master_df = metrics_df
        master_df.to_csv(master_csv, index=False)
    
    return json_file, csv_file if "test_metrics" in results else None


def compare_all_models(cfg):
    """Generate comprehensive model comparison and identify best performing model."""
    master_csv = cfg.METRICS_DIR / "all_model_results.csv"
    
    if not master_csv.exists():
        print("No model results found for comparison.")
        return None
    
    df = pd.read_csv(master_csv)
    
    if df.empty:
        print("No model results available for comparison.")
        return None
    
    # Group by backbone and get latest results for each
    latest_results = df.loc[df.groupby('backbone')['timestamp'].idxmax()]
    
    # Sort by macro_f1 (primary metric)
    if 'macro_f1' in latest_results.columns:
        latest_results = latest_results.sort_values('macro_f1', ascending=False)
        best_model = latest_results.iloc[0]
        
        # Create comparison report
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "best_model": {
                "backbone": best_model['backbone'],
                "macro_f1": float(best_model['macro_f1']) if pd.notna(best_model['macro_f1']) else None,
                "accuracy": float(best_model['accuracy']) if 'accuracy' in best_model and pd.notna(best_model['accuracy']) else None,
                "weighted_f1": float(best_model['weighted_f1']) if 'weighted_f1' in best_model and pd.notna(best_model['weighted_f1']) else None,
                "timestamp": best_model['timestamp']
            },
            "all_models_ranking": []
        }
        
        for idx, row in latest_results.iterrows():
            model_result = {
                "rank": idx + 1,
                "backbone": row['backbone'],
                "macro_f1": float(row['macro_f1']) if pd.notna(row['macro_f1']) else None,
                "accuracy": float(row['accuracy']) if 'accuracy' in row and pd.notna(row['accuracy']) else None,
                "weighted_f1": float(row['weighted_f1']) if 'weighted_f1' in row and pd.notna(row['weighted_f1']) else None,
                "timestamp": row['timestamp']
            }
            comparison["all_models_ranking"].append(model_result)
        
        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = cfg.METRICS_DIR / f"model_comparison_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save ranking as CSV
        ranking_df = pd.DataFrame(comparison["all_models_ranking"])
        ranking_csv = cfg.METRICS_DIR / f"model_ranking_{timestamp}.csv"
        ranking_df.to_csv(ranking_csv, index=False)
        
        print(f"\n{'='*60}")
        print("  MODEL COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Best Model: {best_model['backbone']}")
        print(f"Best Macro-F1: {best_model['macro_f1']:.4f}")
        print(f"Comparison saved to: {comparison_file}")
        print(f"Ranking saved to: {ranking_csv}")
        print(f"{'='*60}\n")
        
        return comparison
    
    else:
        print("No macro_f1 metric found for comparison.")
        return None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Medicinal Plant Leaf Classification Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "train", "evaluate", "ablation", "cv", "gradcam"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Override backbone from config (e.g., resnet50)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild registry even if it exists on disk",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if available",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all trained models and show best performer",
    )
    return parser.parse_args()


def _load_or_build_registry(cfg, rebuild: bool = False):
    """Loads or builds the path-label registry with split assignments."""
    splits_path = cfg.METRICS_DIR / "registry_with_splits.csv"

    if splits_path.exists() and not rebuild:
        import pandas as pd
        print(f"[Main] Loading registry with splits from {splits_path}")
        registry = pd.read_csv(splits_path)
        registry["is_augmented"] = registry["is_augmented"].astype(bool)
    else:
        registry = run_preprocessing(cfg)
        registry = stratified_split_on_originals(
            registry   = registry,
            train_ratio= cfg.TRAIN_RATIO,
            val_ratio  = cfg.VAL_RATIO,
            test_ratio = cfg.TEST_RATIO,
            seed       = cfg.RANDOM_SEED,
        )
        cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        registry.to_csv(splits_path, index=False)
        print(f"[Main] Registry saved to {splits_path}")

    return registry


def stage_preprocess(cfg, rebuild: bool) -> object:
    print("\n[Main] ── Stage: PREPROCESS ──")
    registry = _load_or_build_registry(cfg, rebuild=rebuild)
    print(f"[Main] Registry ready — {len(registry)} entries.")
    return registry


def stage_train(cfg, registry, logger=None, resume_state=None) -> tuple:
    print("\n[Main] ── Stage: TRAIN ──")
    if logger:
        logger.info(f"Starting training stage with backbone: {cfg.BACKBONE}")
    
    loaders, datasets = get_dataloaders(registry, cfg)
    class_weights     = compute_class_weights(registry, cfg.NUM_CLASSES)
    model             = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, cfg.PRETRAINED)
    
    # Check for resumable checkpoint
    start_epoch = 1
    if resume_state and resume_state.get("checkpoint_path"):
        checkpoint_path = Path(resume_state["checkpoint_path"])
        if checkpoint_path.exists():
            print(f"[Main] Resuming training from checkpoint: {checkpoint_path}")
            if logger:
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            
            if start_epoch > cfg.EPOCHS:
                print(f"[Main] Training already completed ({start_epoch-1}/{cfg.EPOCHS} epochs)")
                return model, loaders, datasets, checkpoint.get("training_history", {})
    
    history = train(cfg, model, loaders, class_weights, start_epoch=start_epoch, logger=logger)
    
    # Save training results
    training_results = {
        "backbone": cfg.BACKBONE,
        "training_completed": True,
        "total_epochs": len(history.get("train_loss", [])),
        "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
        "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
        "best_val_macro_f1": max(history["val_macro_f1"]) if history.get("val_macro_f1") else None,
        "training_history": history,
        "timestamp": datetime.now().isoformat()
    }
    
    save_model_results(cfg, cfg.BACKBONE, training_results, "training")
    
    return model, loaders, datasets, history


def stage_evaluate(cfg, model, loaders, logger=None) -> dict:
    print("\n[Main] ── Stage: EVALUATE ──")
    if logger:
        logger.info(f"Starting evaluation stage with backbone: {cfg.BACKBONE}")
    
    metrics = evaluate(
        model       = model,
        dataloader  = loaders["test"],
        class_names = cfg.CLASS_NAMES,
        device      = DEVICE,
        split_name  = "test",
        backbone    = cfg.BACKBONE,
        metrics_dir = cfg.METRICS_DIR,
        figures_dir = cfg.FIGURES_DIR,
    )
    
    # Enhanced results structure
    evaluation_results = {
        "backbone": cfg.BACKBONE,
        "evaluation_completed": True,
        "test_metrics": metrics,
        "dataset_info": {
            "test_samples": len(loaders["test"].dataset),
            "num_classes": cfg.NUM_CLASSES,
            "class_names": cfg.CLASS_NAMES
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save evaluation results
    json_file, csv_file = save_model_results(cfg, cfg.BACKBONE, evaluation_results, "evaluation")
    
    if logger:
        logger.info(f"Evaluation completed. Results saved to: {json_file}")
        if "accuracy" in metrics:
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        if "macro_f1" in metrics:
            logger.info(f"Test Macro-F1: {metrics['macro_f1']:.4f}")
    
    print(f"\n[Evaluation] Results saved to:")
    print(f"  JSON: {json_file}")
    if csv_file:
        print(f"  CSV:  {csv_file}")
    
    return metrics


def stage_gradcam(cfg, model, datasets) -> None:
    print("\n[Main] ── Stage: GRAD-CAM ──")
    visualize_gradcam_grid(
        model              = model,
        dataset            = datasets["test"],
        class_names        = cfg.CLASS_NAMES,
        backbone           = cfg.BACKBONE,
        n_samples_per_class= 3,
        figures_dir        = cfg.FIGURES_DIR,
    )


def stage_ablation(cfg, registry) -> None:
    print("\n[Main] ── Stage: ABLATION ──")
    # E2 is the PRIMARY paper contribution — run first
    experiment_e2_leakage_contamination(cfg, registry)
    experiment_e1_augmentation_value(cfg, registry)
    experiment_e3_class_weighting(cfg, registry)
    experiment_e4_augmentation_types(cfg, registry)
    experiment_e5_transfer_learning(cfg, registry)


def stage_cv(cfg, registry) -> None:
    print("\n[Main] ── Stage: CROSS-VALIDATION ──")
    run_kfold_cv(cfg, registry, backbone=cfg.BACKBONE)
    # Also run for all baselines
    for baseline in cfg.BASELINES:
        print(f"\n[Main] CV for baseline: {baseline}")
        run_kfold_cv(cfg, registry, backbone=baseline)


def _print_completion_summary(cfg) -> None:
    """Prints pipeline completion summary with all saved file paths."""
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {cfg.OUTPUT_DIR.resolve()}\n")

    sections = {
        "Checkpoints": cfg.CHECKPOINT_DIR,
        "Metrics":     cfg.METRICS_DIR,
        "Figures":     cfg.FIGURES_DIR,
    }
    for section, directory in sections.items():
        if directory.exists():
            files = sorted(directory.iterdir())
            print(f"  {section}/")
            for f in files:
                print(f"    {f.name}")
        else:
            print(f"  {section}/  (empty)")
    
    # Show latest model comparison if available
    master_csv = cfg.METRICS_DIR / "all_model_results.csv"
    if master_csv.exists():
        print(f"\n  Model Results Summary:")
        try:
            df = pd.read_csv(master_csv)
            if not df.empty:
                latest_results = df.loc[df.groupby('backbone')['timestamp'].idxmax()]
                if 'macro_f1' in latest_results.columns:
                    latest_results = latest_results.sort_values('macro_f1', ascending=False)
                    print(f"    Best Model: {latest_results.iloc[0]['backbone']} (Macro-F1: {latest_results.iloc[0]['macro_f1']:.4f})")
                    print(f"    Total Models Evaluated: {len(latest_results)}")
        except Exception as e:
            print(f"    Could not load results summary: {e}")
    
    print(f"\n{'='*60}\n")


def main():
    args = _parse_args()

    # Build config object (module as config namespace)
    cfg = cfg_module

    # Override backbone if specified
    if args.backbone:
        if args.backbone not in (["efficientnet_b4"] + cfg.BASELINES):
            print(f"⚠ WARNING: '{args.backbone}' is not in config BASELINES list. Proceeding.")
        cfg.BACKBONE = args.backbone
        print(f"[Main] Backbone overridden to: {cfg.BACKBONE}")

    # Create all output directories
    for d in [cfg.CHECKPOINT_DIR, cfg.METRICS_DIR, cfg.FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger, log_file = setup_logging(cfg, args.stage, cfg.BACKBONE)
    logger.info(f"Pipeline started - Stage: {args.stage}, Backbone: {cfg.BACKBONE}, Device: {DEVICE}")

    print(f"\n[Main] Device: {DEVICE}")
    print(f"[Main] Backbone: {cfg.BACKBONE}")
    print(f"[Main] Stage: {args.stage}")
    print(f"[Main] Logging to: {log_file}")

    # Handle model comparison request
    if args.compare:
        compare_all_models(cfg)
        return

    # Check for resume capability
    resume_state = None
    if args.resume:
        resume_state = load_pipeline_state(cfg, cfg.BACKBONE)
        if resume_state:
            print(f"[Main] Resuming from previous state: {resume_state['timestamp']}")
            logger.info(f"Resuming from state: {resume_state}")
        else:
            print(f"[Main] No previous state found for {cfg.BACKBONE}")
            logger.info(f"No previous state found for {cfg.BACKBONE}")

    # Validate dataset exists early
    if not cfg.ORIGINAL_DIR.exists():
        error_msg = (f"\n🔴 Dataset not found at: {cfg.ORIGINAL_DIR}\n"
                    f"Please place your dataset under: {cfg.DATA_DIR}\n"
                    f"Expected structure:\n"
                    f"  data/dataset/Original Images/<ClassName>/\n"
                    f"  data/dataset/Augmented Images/<ClassName>/\n")
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Initialize pipeline state tracking
    pipeline_state = {
        "completed_stages": resume_state.get("completed_stages", []) if resume_state else [],
        "current_epoch": 0,
        "best_metrics": {},
        "training_history": {},
        "checkpoint_path": ""
    }

    # ── Pipeline execution ────────────────────────────────────────────────────
    registry     = None
    model        = None
    loaders      = None
    datasets_map = None

    try:
        # Stage: Preprocessing
        if args.stage in ("all", "preprocess"):
            if "preprocess" not in pipeline_state["completed_stages"]:
                registry = stage_preprocess(cfg, rebuild=args.rebuild)
                pipeline_state["completed_stages"].append("preprocess")
                save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
                logger.info("Preprocessing stage completed")
            else:
                print("[Main] Preprocessing already completed (resuming)")
                logger.info("Preprocessing stage skipped (already completed)")

        if registry is None:
            registry = _load_or_build_registry(cfg, rebuild=False)

        # Stage: Training
        if args.stage in ("all", "train"):
            if "train" not in pipeline_state["completed_stages"]:
                model, loaders, datasets_map, history = stage_train(cfg, registry, logger, resume_state)
                pipeline_state["completed_stages"].append("train")
                pipeline_state["training_history"] = history
                pipeline_state["checkpoint_path"] = str(cfg.CHECKPOINT_DIR / f"{cfg.BACKBONE}_best.pth")
                save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
                logger.info("Training stage completed")
            else:
                print("[Main] Training already completed (resuming)")
                logger.info("Training stage skipped (already completed)")

        # Stage: Evaluation
        if args.stage in ("all", "evaluate"):
            if model is None:
                # Load best checkpoint
                ckpt_path = cfg.CHECKPOINT_DIR / f"{cfg.BACKBONE}_best.pth"
                if not ckpt_path.exists():
                    error_msg = f"No checkpoint found at {ckpt_path}. Run --stage train first."
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                print(f"[Main] Loading checkpoint: {ckpt_path}")
                logger.info(f"Loading checkpoint: {ckpt_path}")
                model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, pretrained=False)
                ckpt  = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(DEVICE)

            if loaders is None:
                loaders, datasets_map = get_dataloaders(registry, cfg)

            metrics = stage_evaluate(cfg, model, loaders, logger)
            pipeline_state["best_metrics"] = metrics
            pipeline_state["completed_stages"].append("evaluate")
            save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
            logger.info("Evaluation stage completed")

        # Stage: Grad-CAM
        if args.stage in ("all", "gradcam"):
            if model is None:
                ckpt_path = cfg.CHECKPOINT_DIR / f"{cfg.BACKBONE}_best.pth"
                if not ckpt_path.exists():
                    error_msg = f"No checkpoint found at {ckpt_path}. Run --stage train first."
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                model = get_model(cfg.BACKBONE, cfg.NUM_CLASSES, pretrained=False)
                ckpt  = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(DEVICE)

            if datasets_map is None:
                _, datasets_map = get_dataloaders(registry, cfg)

            stage_gradcam(cfg, model, datasets_map)
            pipeline_state["completed_stages"].append("gradcam")
            save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
            logger.info("Grad-CAM stage completed")

        # Stage: Ablation Studies
        if args.stage in ("all", "ablation"):
            stage_ablation(cfg, registry)
            pipeline_state["completed_stages"].append("ablation")
            save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
            logger.info("Ablation studies completed")

        # Stage: Cross-Validation
        if args.stage in ("all", "cv"):
            stage_cv(cfg, registry)
            pipeline_state["completed_stages"].append("cv")
            save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
            logger.info("Cross-validation completed")

        # Generate model comparison at the end
        if args.stage == "all":
            print("\n[Main] Generating model comparison...")
            compare_all_models(cfg)
            logger.info("Model comparison generated")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        # Save current state even if failed
        save_pipeline_state(cfg, args.stage, cfg.BACKBONE, pipeline_state)
        raise

    _print_completion_summary(cfg)
    logger.info("Pipeline completed successfully")
    
    # Close logging handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()
