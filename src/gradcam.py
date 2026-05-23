# src/gradcam.py — Grad-CAM + ViT attention map visualization
# pip install torch torchvision numpy matplotlib Pillow

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fraction of image border that counts as "edge" for background detection heuristic
BORDER_FRACTION = 0.10


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    backbone: str,
) -> np.ndarray:
    """
    Generates Grad-CAM heatmap for a single image and target class.

    CNNs (EfficientNet, ResNet, VGG, MobileNet):
      - Hook last conv layer; compute gradient of class score w.r.t. activations.
      - Weight feature maps by GAP of gradients; ReLU + upsample.

    ViT:
      - Attention rollout: average attention across all heads/layers.

    Returns heatmap as numpy array (H, W) normalized to [0, 1].
    """
    if "vit" in backbone.lower():
        return _vit_attention_rollout(model, image_tensor)
    else:
        return _gradcam_cnn(model, image_tensor, target_class, backbone)


def _get_last_conv_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Returns the last convolutional layer of a CNN backbone."""
    if backbone == "efficientnet_b4":
        # Last conv block in features
        return model.features[-1][0]
    elif backbone == "resnet50":
        return model.layer4[-1].conv3
    elif backbone == "vgg16":
        return model.features[-3]   # Last conv in VGG16 features
    elif backbone == "mobilenet_v3_large":
        return model.features[-1][0]
    elif backbone == "swin_t":
        # torchvision Swin uses model.permute (Permute module) after model.norm.
        # Its output is (B, C, H, W) — correct BCHW format for Grad-CAM.
        # Do NOT use model.layers (timm naming); torchvision uses model.features/permute.
        return model.permute
    else:
        raise ValueError(f"Unknown CNN backbone for Grad-CAM: {backbone}")


def _gradcam_cnn(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    backbone: str,
) -> np.ndarray:
    """Grad-CAM for CNN models via forward/backward hooks on last conv layer."""
    model.eval()
    activations = {}
    gradients   = {}

    last_conv = _get_last_conv_layer(model, backbone)

    def fwd_hook(module, input, output):
        activations["value"] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        if grad_out[0] is not None:
            gradients["value"] = grad_out[0].detach()

    fwd_handle = last_conv.register_forward_hook(fwd_hook)
    bwd_handle = last_conv.register_full_backward_hook(bwd_hook)

    inp = image_tensor.unsqueeze(0).to(DEVICE)
    inp.requires_grad_(True)  # Must require gradients for backward pass

    output = model(inp)
    score  = output[0, target_class]

    model.zero_grad()
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Check if hooks were triggered
    if "value" not in activations or "value" not in gradients:
        print(f"[Grad-CAM] Warning: Hooks not triggered for {backbone}. Returning uniform heatmap.")
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return np.ones((H, W)) * 0.5

    acts  = activations["value"].squeeze(0)   # (C, H, W)
    grads = gradients["value"].squeeze(0)     # (C, H, W)

    # GAP over spatial dims to get per-channel weights
    weights = grads.mean(dim=(1, 2))          # (C,)

    # Weighted sum of feature maps - ensure cam is on same device as acts
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)   # Keep only positive activations

    # Upsample to input size with bilinear interpolation
    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam


def _vit_attention_rollout(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """
    ViT attention rollout: averages attention across all heads/layers.
    Returns heatmap of shape (H, W) normalized to [0, 1].
    """
    model.eval()
    attentions = []

    # Collect attention weights via monkey-patching encoder blocks
    original_forwards = []
    for blk in model.encoder.layers:
        original_forwards.append(blk.self_attention.forward)

        def make_hook(block):
            def hooked_forward(query, key, value, **kwargs):
                # Force return of attention weights
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, attn = nn.MultiheadAttention.forward(
                    block.self_attention, query, key, value, **kwargs
                )
                attentions.append(attn.detach().cpu())
                return out, attn
            return hooked_forward

        blk.self_attention.forward = make_hook(blk)

    inp = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _ = model(inp)

    # Restore original forwards
    for blk, orig_fwd in zip(model.encoder.layers, original_forwards):
        blk.self_attention.forward = orig_fwd

    if not attentions:
        # Fallback: return uniform heatmap
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return np.ones((H, W)) * 0.5

    # Attention rollout algorithm
    # Each attention: (batch, num_heads, seq_len, seq_len) - already on CPU from hook
    rollout = torch.eye(attentions[0].shape[-1], device='cpu')  # Explicitly on CPU since attentions are on CPU
    for attn in attentions:
        attn_avg = attn[0].mean(dim=0)              # Average over heads
        attn_avg = attn_avg + torch.eye(attn_avg.shape[0], device='cpu')  # Residual connection - both on CPU
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        rollout  = torch.matmul(attn_avg, rollout)

    # CLS token attends to patches; take row 0 (CLS → patches)
    cls_attn = rollout[0, 1:]   # Exclude CLS token itself
    grid_size = int(cls_attn.shape[0] ** 0.5)
    cls_attn  = cls_attn.reshape(grid_size, grid_size).numpy()

    # Upsample to input size
    # Normalize to [0, 255] uint8 before PIL for cross-platform safety
    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    if cls_attn.max() > cls_attn.min():
        cls_attn_u8 = ((cls_attn - cls_attn.min()) /
                       (cls_attn.max() - cls_attn.min()) * 255).astype(np.uint8)
    else:
        cls_attn_u8 = np.zeros_like(cls_attn, dtype=np.uint8)
    heatmap = Image.fromarray(cls_attn_u8, mode="L").resize((W, H), Image.BILINEAR)
    heatmap = np.array(heatmap, dtype=np.float32) / 255.0   # Back to [0, 1]
    return heatmap


def _check_background_attention(heatmap: np.ndarray) -> bool:
    """
    Background detection heuristic:
    If the centroid of Grad-CAM activation falls within 10% of image border → flag.
    Returns True if likely background attention (concern for paper).
    """
    H, W    = heatmap.shape
    ys, xs  = np.indices((H, W))
    total   = heatmap.sum()

    if total < 1e-8:
        return False

    cy = float((ys * heatmap).sum() / total)   # Centroid y
    cx = float((xs * heatmap).sum() / total)   # Centroid x

    border_y = BORDER_FRACTION * H
    border_x = BORDER_FRACTION * W

    return (
        cy < border_y or cy > H - border_y or
        cx < border_x or cx > W - border_x
    )


def _overlay_heatmap(orig_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blends Grad-CAM heatmap (jet colormap) over original image."""
    colormap = cm_module.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[:, :, :3]   # Drop alpha
    overlay = 0.5 * orig_img.astype(np.float32) / 255.0 + 0.5 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def visualize_gradcam_grid(
    model: nn.Module,
    dataset,
    class_names: list,
    backbone: str,
    n_samples_per_class: int = 3,
    output_path: Path = None,
    figures_dir: Path = None,
) -> None:
    """
    Produces a (6 × n_samples) grid figure of Grad-CAM overlays.

    For each class: selects n_samples_per_class correctly classified test images.
    Saves to figures_dir/gradcam_grid_<backbone>.png.

    Flags background attention with a visible warning (required paper finding).
    """  # PRD §11
    model.eval()
    num_classes = len(class_names)

    # Collect correctly classified samples per class
    per_class_samples = {c: [] for c in range(num_classes)}

    for idx in range(len(dataset)):
        if all(len(v) >= n_samples_per_class for v in per_class_samples.values()):
            break

        img_tensor, label = dataset[idx]
        true_class = int(label)

        if len(per_class_samples[true_class]) >= n_samples_per_class:
            continue

        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(DEVICE))
            pred   = int(logits.argmax(dim=1).item())

        if pred == true_class:
            per_class_samples[true_class].append(img_tensor)

    n_cols = n_samples_per_class
    n_rows = num_classes
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, cls_idx in enumerate(range(num_classes)):
        cls_name = class_names[cls_idx]
        samples  = per_class_samples[cls_idx]

        for col, img_tensor in enumerate(samples[:n_cols]):
            # Original image (de-normalize for display)
            orig_np = _denormalize_tensor(img_tensor)

            # Grad-CAM heatmap
            heatmap = generate_gradcam(model, img_tensor, cls_idx, backbone)

            # Background attention check  # PRD §11.1
            if _check_background_attention(heatmap):
                print(f"⚠ WARNING: Grad-CAM background attention detected "
                      f"for class '{cls_name}' (sample {col}). "
                      f"Model may be attending to background, not leaf morphology.")

            overlay = _overlay_heatmap(orig_np, heatmap)

            # Original
            ax_orig = axes[row, col * 2]
            ax_orig.imshow(orig_np)
            ax_orig.axis("off")
            if col == 0:
                ax_orig.set_ylabel(cls_name, fontsize=9, rotation=90, va="center")
            ax_orig.set_title("Original", fontsize=7)

            # Overlay
            ax_ov = axes[row, col * 2 + 1]
            ax_ov.imshow(overlay)
            ax_ov.axis("off")
            ax_ov.set_title("Grad-CAM", fontsize=7)

        # Fill empty slots if fewer than n_samples_per_class
        for col in range(len(samples), n_cols):
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

    plt.suptitle(f"Grad-CAM Visualizations — {backbone}", fontsize=12, y=1.01)
    plt.tight_layout()

    if output_path is None and figures_dir is not None:
        output_path = figures_dir / f"gradcam_grid_{backbone}.png"
    elif output_path is None:
        output_path = Path("gradcam_grid.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] Grid saved to {output_path}")


def _denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # (C, H, W) → (H, W, C)
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
