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
QUERY_EMBED_ROOT = Path("/exports/dinov3_embed")
QUERY_VIS_ROOT = Path("/exports/dinov3_vis")
SUBDIRS: Sequence[Path] = (
    Path("Q250912150549_400"),
    Path("Q250912154506_300"),
    Path("Q250912161658_200"),
)

GRID_PATTERN = "QueryPatchGrid_*.npy"


def iter_grid_files(base_dir: Path) -> Iterable[Path]:
    if SUBDIRS:
        dirs = [base_dir / pattern for pattern in SUBDIRS]
    else:
        dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    for target_dir in dirs:
        if not target_dir.exists():
            print(f"[WARN] Query embed subdir missing, skipping: {target_dir}")
            continue
        for path in sorted(target_dir.glob(GRID_PATTERN)):
            yield path


def save_dense_feature(grid_path: Path, output_path: Path) -> None:
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
    print(f"[saved] DenseFT -> {output_path}")


def main() -> None:
    embed_base = QUERY_EMBED_ROOT / f"Q{VAR_WEIGHT_KEY}"
    vis_base = QUERY_VIS_ROOT / f"Q{VAR_WEIGHT_KEY}"

    total = 0
    for grid_path in iter_grid_files(embed_base):
        rel = grid_path.relative_to(embed_base)
        output_path = vis_base / rel.with_name(rel.stem.replace("QueryPatchGrid", "QueryDenseFT") + ".png")
        save_dense_feature(grid_path, output_path)
        total += 1

    print(f"[DONE] Generated {total} dense feature images for queries.")


if __name__ == "__main__":
    main()
