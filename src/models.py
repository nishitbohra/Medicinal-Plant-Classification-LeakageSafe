# src/models.py — All model architectures
# pip install torch torchvision

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision.models as tvm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _freeze_backbone(model: nn.Module) -> None:
    """Freezes all parameters except the final classification head."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all model parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def _print_param_summary(model: nn.Module, backbone: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {backbone}: total_params={total:,}  trainable_params={trainable:,}")


def get_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Factory function returning a modified torchvision model.

    Supported backbones:
      'efficientnet_b4'    — PRIMARY (torchvision)
      'resnet50'           — Residual CNN baseline
      'vgg16'              — Historical baseline
      'mobilenet_v3_large' — Mobile/edge baseline
      'vit_b_16'           — Pure Transformer baseline
      'swin_t'             — Hierarchical Transformer baseline

    All models:
      - Final head replaced with Linear(in_features, num_classes).
      - Backbone frozen initially; unfreeze after 5 warm-up epochs via unfreeze_backbone().
      - Pretrained ImageNet weights loaded when pretrained=True.

    Returns model moved to correct device.
    """
    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "efficientnet_b4":
        model = tvm.efficientnet_b4(weights=weights_arg)
        _freeze_backbone(model)
        # Replace classifier[1] with Linear(1792, num_classes)  # PRD §8.1
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model.classifier[1].requires_grad_(True)   # Always train the head

    elif backbone == "resnet50":
        model = tvm.resnet50(weights=weights_arg)
        _freeze_backbone(model)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.fc.requires_grad_(True)

    elif backbone == "vgg16":
        model = tvm.vgg16(weights=weights_arg)
        _freeze_backbone(model)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        model.classifier[6].requires_grad_(True)

    elif backbone == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=weights_arg)
        _freeze_backbone(model)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        model.classifier[3].requires_grad_(True)

    elif backbone == "vit_b_16":
        model = tvm.vit_b_16(weights=weights_arg)
        _freeze_backbone(model)
        # Replace heads.head with Linear(768, num_classes)  # PRD §8.2
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        model.heads.head.requires_grad_(True)

    elif backbone == "swin_t":
        model = tvm.swin_t(weights=weights_arg)
        _freeze_backbone(model)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        model.head.requires_grad_(True)

    else:
        raise ValueError(
            f"Unknown backbone: '{backbone}'. "
            f"Choose from: efficientnet_b4, resnet50, vgg16, "
            f"mobilenet_v3_large, vit_b_16, swin_t"
        )

    model = model.to(DEVICE)
    _print_param_summary(model, backbone)
    return model


def unfreeze_backbone(model: nn.Module, backbone: str) -> None:
    """
    Unfreezes all backbone layers after the warm-up phase (epoch 5).
    Called from the training loop at the end of epoch 5.
    """
    _unfreeze_backbone(model)
    _print_param_summary(model, backbone)
    print(f"[Model] Backbone '{backbone}' unfrozen — all layers now trainable.")
