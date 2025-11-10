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
    weights_path,
    file_prefix
)

varAltitude = 450
varIndex = 1
varWeight = "vit7b16"


def _build_context(
    altitude: int,
    index: int,
    weight: str,
    variant: str,
    embedding_cfg: str,
) -> dict[str, Path | str]:
    """Prepare the shared paths used throughout the dense feature pipeline."""
    hub_entry, _, dataset_type = weights_path(weight) # e.g. dinov3_vitl16
    altitude_str = f"{int(altitude)}"
    index_str = f"{int(index):04d}"
    prefix = file_prefix(altitude, index) # e.g. 200_0150

    grid_name = f"PatchGrid_{embedding_cfg}_{variant}_{hub_entry}_{dataset_type}_{altitude_str}_{index_str}"
    dense_name = f"DenseFT_{embedding_cfg}_{variant}_{hub_entry}_{dataset_type}_{altitude_str}_{index_str}"

    altitude_dir = EMBED_ROOT / weight / altitude_str
    grid_dir = altitude_dir / "PatchGrid"
    dense_dir = altitude_dir / "DenseFT"

    return {
        "hub_entry": hub_entry,
        "prefix": prefix,
        "grid_path": grid_dir / f"{grid_name}.npy",
        "dense_path": dense_dir / f"{dense_name}.png",
        "dense_dir": dense_dir,
    }


def generate_dense_feature(
    altitude: int,
    index: int,
    weight: str,
    target_res: int = 1024,
    variant: str = "raw",
    embedding_cfg: str | None = None,
) -> None:
    """Load the patch grid exported by Test_global_embedding and save a PNG."""
    resolved_embedding_cfg = embedding_cfg or f"res{int(target_res)}_ImageNet"
    ctx = _build_context(altitude, index, weight, variant, resolved_embedding_cfg)
    grid_path = ctx["grid_path"]
    dense_path = ctx["dense_path"]
    dense_dir = ctx["dense_dir"]

    if not grid_path.exists():
        raise FileNotFoundError(
            f"\033[91mPatch grid not found for altitude={altitude}, index={index}, weight={weight}: {grid_path}\033[0m"
        )

    Path(dense_dir).mkdir(parents=True, exist_ok=True)

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
    print(f"\033[32m[saved] Dense feature image -> {dense_path}\033[0m")


def main() -> None:
    generate_dense_feature(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
    )
    print(f"\033[32m[DONE] Generated dense feature image.\033[0m")


if __name__ == "__main__":
    main()
