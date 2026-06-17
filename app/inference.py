"""Model loading, TTA inference, Grad-CAM, and heatmap overlay.

The V2-S builder must mirror the training-time _build_efficientnet_v2_s in the
notebook (same head structure, same stochastic_depth_prob) so the trained
.pth loads cleanly. Likewise, get_clean_transform / get_tta_transform must
match the notebook's get_val_transforms / get_tta_transforms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────
# Model builder — only EfficientNetV2-S is shipped to production. The
# head MUST mirror the notebook's _build_efficientnet_v2_s exactly so the
# state_dict loads cleanly.
# ─────────────────────────────────────────────────────────────────────

def _build_efficientnet_v2_s(num_classes):
    # stochastic_depth_prob=0.3 documents the trained configuration. DropPath
    # is bypassed in eval mode, so the value does not affect inference — it's
    # set here so the architecture exactly matches the training-time builder.
    model = models.efficientnet_v2_s(
        weights=None,
        stochastic_depth_prob=0.3,
    )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


_BUILDER = {
    'efficientnet_v2_s': _build_efficientnet_v2_s,
}


def load_model(model_name, weights_path, num_classes=7):
    """Build the architecture, load .pth weights, send to device, eval mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _BUILDER[model_name](num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ─────────────────────────────────────────────────────────────────────
# Transforms — must match the notebook val + TTA transforms exactly.
# ─────────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_clean_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(img_size * 1.07)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def get_tta_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=45, shear=(-10, 10)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ─────────────────────────────────────────────────────────────────────
# Inference: TTA averaging with temperature-calibrated softmax.
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_tta(model, image_hwc_uint8, img_size, device,
            n_augmented=8, temperature: float = 1.0):
    """Average softmax across (1 clean + n_augmented) views, calibrated.

    Args:
        image_hwc_uint8: HxWxC RGB uint8 numpy array (the original upload).
        n_augmented:     Number of augmented TTA passes.
        temperature:     Calibration scalar — logits are divided by this
                         before softmax. 1.0 disables calibration.

    Returns:
        np.ndarray of shape (num_classes,) — averaged calibrated probabilities.
    """
    model.eval()
    clean_t = get_clean_transform(img_size)
    aug_t   = get_tta_transform(img_size)

    clean_x = clean_t(image_hwc_uint8).unsqueeze(0).to(device)
    prob_sum = (
        F.softmax(model(clean_x) / temperature, dim=1)[0]
        .cpu().numpy().astype(np.float64)
    )

    for _ in range(n_augmented):
        aug_x = aug_t(image_hwc_uint8).unsqueeze(0).to(device)
        prob_sum += (
            F.softmax(model(aug_x) / temperature, dim=1)[0]
            .cpu().numpy().astype(np.float64)
        )

    return prob_sum / (1 + n_augmented)


# ─────────────────────────────────────────────────────────────────────
# Grad-CAM (Selvaraju et al.) on the clean center-crop view.
# Temperature scaling is NOT applied here — Grad-CAM uses gradients of
# the logit, which are invariant to a positive scalar division.
# ─────────────────────────────────────────────────────────────────────

def compute_gradcam(model, image_hwc_uint8, img_size, device,
                    target_layer, target_class_idx):
    """Forward+backward over the clean view, weight activations by gradients.

    Returns:
        2D float32 numpy array, normalized to [0, 1], at the resolution of
        target_layer's output. Caller is responsible for resizing to image.
    """
    model.eval()
    clean_t = get_clean_transform(img_size)
    x = clean_t(image_hwc_uint8).unsqueeze(0).to(device).requires_grad_(True)

    activations = {}
    gradients = {}

    def fwd_hook(module, inputs, output):
        # Keep output in the autograd graph so backward can flow through it.
        activations['value'] = output

    def bwd_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(x)
        score = logits[0, target_class_idx]
        model.zero_grad()
        score.backward()

        a = activations['value'][0]    # (C, H, W)
        g = gradients['value'][0]      # (C, H, W)
        weights = g.mean(dim=(1, 2))   # (C,) — global-avg-pooled gradients
        cam = torch.relu((weights[:, None, None] * a).sum(dim=0))  # (H, W)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()
    finally:
        fwd_h.remove()
        bwd_h.remove()


# ─────────────────────────────────────────────────────────────────────
# Overlay: alpha-blend an RdYlBu_r-colored CAM over the original image.
# ─────────────────────────────────────────────────────────────────────

def build_overlay(original_pil, cam_2d, alpha=0.45):
    """Resize the CAM to the original image, apply the colormap, alpha-blend.

    Args:
        original_pil: PIL.Image in RGB.
        cam_2d:       float32 numpy array in [0,1], any spatial size.
    Returns:
        PIL.Image with the heatmap overlay.
    """
    W, H = original_pil.size
    cam_resized = np.array(
        Image.fromarray((cam_2d * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ) / 255.0

    # RdYlBu reversed: navy → cyan → green → yellow → orange → red.
    # Matches the legend gradient shown in the UI.
    # `matplotlib.colormaps[...]` is the post-3.9 replacement for the removed
    # `matplotlib.cm.get_cmap(...)` — same return type (a Colormap instance) since HF is running on 3.11.
    cmap = matplotlib.colormaps['RdYlBu_r']
    heat_rgb = (cmap(cam_resized)[..., :3] * 255).astype(np.uint8)

    orig_rgb = np.array(original_pil.convert('RGB'))
    blended = ((1 - alpha) * orig_rgb + alpha * heat_rgb).astype(np.uint8)
    return Image.fromarray(blended)
