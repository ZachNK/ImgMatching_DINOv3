"""
Generate dense feature visualisations from query patch grids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


VAR_WEIGHT_KEY = "vitb16"
QUERY_EMBED_ROOT = Path("/exports/dinov3_query_embeds")
ALTITUDE_FILTER: Sequence[int] = ()

GRID_PATTERN = "QueryPatchGrid_*.npy"


def iter_grid_files(
    weight_key: str,
    root: Path = QUERY_EMBED_ROOT,
    altitudes: Sequence[int] | None = None,
) -> Iterable[Path]:
    weight_root = root / weight_key
    if not weight_root.exists():
        print(f"\033[91m[WARN] Query embed root missing for weight={weight_key}: {weight_root}\033[0m")
        return

    if altitudes:
        altitude_dirs = [weight_root / f"{int(alt)}" for alt in altitudes]
    else:
        altitude_dirs = [p for p in weight_root.iterdir() if p.is_dir()]

    for altitude_dir in altitude_dirs:
        if not altitude_dir.exists():
            print(f"\033[93m[WARN] Altitude directory missing, skipping: {altitude_dir}\033[0m")
            continue
        rotation_dirs = [p for p in altitude_dir.iterdir() if p.is_dir()]
        for rotation_dir in rotation_dirs:
            patch_dir = rotation_dir / "PatchGrid"
            if not patch_dir.exists():
                print(f"\033[93m[WARN] PatchGrid directory missing, skipping: {patch_dir}\033[0m")
                continue
            for path in sorted(patch_dir.glob(GRID_PATTERN)):
                yield path


def derive_dense_output_path(grid_path: Path) -> Path:
    dense_dir = grid_path.parent.parent / "DenseFT"
    dense_name = grid_path.stem.replace("PatchGrid", "DenseFT")
    return dense_dir / f"{dense_name}.png"


def save_dense_feature(grid_path: Path, output_path: Path) -> Path:
    grid = torch.from_numpy(np.load(grid_path))  # (H, W, C)
    flat = grid.reshape(-1, grid.shape[-1])
    feat = flat - flat.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(feat, q=3)
    proj = feat @ v[:, :3]

    rgb = proj.reshape(grid.shape[0], grid.shape[1], 3).numpy()
    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-6)

    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    rgb_up = F.interpolate(rgb, size=(1024, 1024), mode="bilinear", align_corners=False)
    rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb_up * 255).astype("uint8")).save(output_path)
    print(f"[saved] DenseFT -> \033[32m{output_path}\033[0m")
    return output_path


def generate_query_dense_feature(
    grid_path: Path,
    output_path: Path | None = None,
) -> Path:
    target_path = output_path or derive_dense_output_path(grid_path)
    return save_dense_feature(grid_path, target_path)


def main() -> None:
    altitudes = ALTITUDE_FILTER if ALTITUDE_FILTER else None
    total = 0
    for grid_path in iter_grid_files(VAR_WEIGHT_KEY, QUERY_EMBED_ROOT, altitudes):
        generate_query_dense_feature(grid_path)
        total += 1

    print(f"\033[31m[DONE] Generated {total} dense feature images for queries.\033[0m")


if __name__ == "__main__":
    main()
