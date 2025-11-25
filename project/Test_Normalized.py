"""
Small helper to run the existing preprocessing (including normalization)
on a single image and save a visualizable preview under /exports.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from imatch.preprocess import build_transform

# Hardcoded backbone keyword we want to use (must exist in data_key.json).
# Keys correspond to the innermost entries under weights->dinov3_weights->*.
BACKBONE_KEY = "vitb16"

# Paths for metadata and weights.
DATA_KEY_PATH = Path("/workspace/project") / "json" / "data_key.json"
WEIGHTS_DIR = Path("/workspace/weights")


def _get_norm_params(normalize: str):
    """Return mean/std used by build_transform for reversing normalization."""
    if normalize == "LVD":
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean = torch.tensor([0.430, 0.411, 0.296])
        std = torch.tensor([0.213, 0.156, 0.143])
    return mean.view(-1, 1, 1), std.view(-1, 1, 1)


def _lookup_backbone(backbone_key: str):
    """Read data_key.json and return (model_id, weight_file, norm_tag)."""
    data = json.loads(DATA_KEY_PATH.read_text(encoding="utf-8"))
    weights_root = data["weights"]["dinov3_weights"]
    for _, entries in weights_root.items():
        if backbone_key in entries:
            model_id, weight_file, norm_tag = entries[backbone_key]
            return model_id, weight_file, norm_tag
    raise KeyError(f"Backbone key '{backbone_key}' not found in {DATA_KEY_PATH}")


def save_normalized_preview(
    image_path: Path,
    output_dir: Path = Path("/exports"),
    patch_size: int = 14,
    patch_multiple: int = 16,
    interpolation: str = "bicubic",
):
    """
    Apply build_transform (including normalization) and save a JPEG preview.
    JPEG requires data in [0, 1], so we reverse the normalization only for saving.
    Backbone/normalization are auto-selected from data_key.json via BACKBONE_KEY.
    """
    model_id, weight_file, norm_tag = _lookup_backbone(BACKBONE_KEY)
    weight_path = WEIGHTS_DIR / weight_file
    normalize = "LVD" if norm_tag.upper() == "LVD" else "vits16+"
    print(
        f"[preview] backbone_key={BACKBONE_KEY}, model_id={model_id}, "
        f"weight={weight_path}, normalize={normalize}"
    )
    if not weight_path.exists():
        print(f"[preview][warn] Weight file not found at {weight_path} (lookup succeeded).")
    img = Image.open(image_path).convert("RGB")
    # build_transform expects a tensor input (it starts with ConvertImageDtype),
    # so convert PIL -> Tensor before applying the transform pipeline.
    img_tensor = transforms.ToTensor()(img)
    transform = build_transform(
        patch_size=patch_size,
        patch_multiple=patch_multiple,
        interpolation=interpolation,
        normalize=normalize,
    )
    normalized = transform(img_tensor)

    mean, std = _get_norm_params(normalize)
    preview = (normalized * std + mean).clamp(0, 1)  # undo normalization for display

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Normalization_{image_path.name}"
    transforms.ToPILImage()(preview).save(output_path, quality=95)
    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a normalized image preview produced by build_transform."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the source image file.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Patch size passed to build_transform.",
    )
    parser.add_argument(
        "--patch-multiple",
        type=int,
        default=16,
        help="Patch multiple passed to build_transform.",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        help="Interpolation mode passed to build_transform.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_normalized_preview(
        image_path=args.image,
        output_dir=Path("/exports"),
        patch_size=args.patch_size,
        patch_multiple=args.patch_multiple,
        interpolation=args.interpolation,
    )
