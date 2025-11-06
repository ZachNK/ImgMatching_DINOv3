"""
Generate dense feature visualisations from the exported patch grids.

The logic is wrapped in generate_dense_feature so it can be reused in batch
runs while keeping the original single-run behaviour.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from imatch.loading import (
    EMBED_ROOT,
    VIS_ROOT,
    weights_path, 
    file_prefix
)

varAltitude = 450
varIndex = 1
varWeight = "vit7b16"

_EXPORT_SUBDIR = EMBED_ROOT / varWeight # e.g. /exports/dinov3_embeds/vit7b16sat
_DENSE_SUBDIR = VIS_ROOT / varWeight # e.g. /exports/dinov3_vis/vit7b16sat


def _build_context(altitude: int, index: int, weight: str) -> dict[str, Path | str]:
    """Prepare the shared paths used throughout the dense feature pipeline."""
    hub_entry, _, dataset_type = weights_path(weight) # e.g. dinov3_vitl16, /opt/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth, SAT
    prefix = file_prefix(altitude, index) # e.g. 200_0150
    export_root = _EXPORT_SUBDIR  # e.g. /exports/dinov3_embeds/vit7b16sat
    dense_root = _DENSE_SUBDIR  # e.g. /exports/dinov3_vis/vit7b16sat

    return {
        "hub_entry": hub_entry, # e.g. "dinov3_vitl16"
        "prefix": prefix, # e.g. "200_0150"
        "grid_path": export_root / f"Global_grid_{hub_entry}_{dataset_type}_{prefix}.npy", # e.g. /exports/dinov3_embeds/vit7b16sat/Global_grid_dinov3_vitl16_SAT_200_0150.npy
        "dense_path": dense_root / f"Dense_Global_{hub_entry}_{dataset_type}_{prefix}.png", # e.g. /exports/dinov3_vis/vit7b16sat/Dense_Global_dinov3_vitl16_SAT_200_0150.png
    }


def generate_dense_feature(altitude: int, index: int, weight: str) -> None:
    """Load the patch grid exported by Test_global_embedding and save a PNG."""
    ctx = _build_context(altitude, index, weight)
    grid_path = ctx["grid_path"]
    dense_path = ctx["dense_path"]

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Patch grid not found for altitude={altitude}, index={index}, weight={weight}: {grid_path}"
        )

    dense_path.parent.mkdir(parents=True, exist_ok=True)

    grid = torch.from_numpy(np.load(grid_path))  # (H, W, C)
    flat = grid.reshape(-1, grid.shape[-1])  # (H*W, C)

    feat = flat - flat.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(feat, q=3)   # v: (C, 3)

    proj = feat @ v[:, :3]                   # (H*W, 3)

    rgb = proj.reshape(grid.shape[0], grid.shape[1], 3).numpy()
    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-6)

    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    rgb_up = F.interpolate(rgb, size=(1024, 1024), mode="bilinear", align_corners=False)
    rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).numpy()

    img = Image.fromarray((rgb_up * 255).astype("uint8"))
    img.save(dense_path)
    print(f"[saved] Dense feature image -> {dense_path}")


def main() -> None:
    generate_dense_feature(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
    )


if __name__ == "__main__":
    main()
