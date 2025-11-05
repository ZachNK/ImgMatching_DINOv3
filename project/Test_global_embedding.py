"""
Utility to extract DINOv3 global embeddings and patch tokens.

This module now exposes run_test_global_embedding so the pipeline can be reused
programmatically while keeping backwards compatibility with the previous CLI
behaviour.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from imatch.features import (
    extract_global_feature,
    extract_patch_tokens,
    reshape_patch_tokens_to_grid,
)
from imatch.io_images import load_image_tensor
from imatch.models import load_model
from imatch.paths import (
    DATASET_ROOT,
    EXPORT_ROOT,
    ckpt_path,
    file_prefix,
    img_path,
)
from imatch.tfms import build_transform
from imatch.utils import progress_bar

# Default parameters kept for manual single-run usage.
varAltitude = 100
varIndex = 1
varWeight = "vits16"
varTargetRes = 1024

REPO_DIR = Path("/workspace/dinov3")
_EXPORT_SUBDIR = Path("dinov3_debug/Test_global_embedding+dense_feature/1106")


def _build_context(altitude: int, index: int, weight: str) -> Dict[str, object]:
    """Assemble frequently reused values for a single inference run."""
    hub_entry, ckpt = ckpt_path(weight)
    img_dir_a, img_dir_b = img_path(altitude, index)
    prefix = file_prefix(altitude, index)

    return {
        "hub_entry": hub_entry,
        "ckpt_path": ckpt,
        "image_path": DATASET_ROOT / f"{img_dir_a}/{img_dir_b}.jpg",
        "file_name": f"GF_{hub_entry}_{prefix}",
        "grid_name": f"GF_grid_{hub_entry}_{prefix}",
        "export_dir": EXPORT_ROOT / _EXPORT_SUBDIR,
    }


def run_test_global_embedding(
    altitude: int,
    index: int,
    weight: str,
    target_res: int = 1024,
) -> None:
    """Execute the full embedding pipeline for the given parameters."""
    ctx = _build_context(altitude, index, weight)
    hub_entry = ctx["hub_entry"]
    ckpt = ctx["ckpt_path"]
    image_path = ctx["image_path"]
    file_name = ctx["file_name"]
    grid_name = ctx["grid_name"]
    export_dir = ctx["export_dir"]

    export_dir.mkdir(parents=True, exist_ok=True)
    npy_path = export_dir / f"{file_name}.npy"
    csv_path = export_dir / f"{file_name}.csv"
    grid_path = export_dir / f"{grid_name}.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "\n================= Debug: Test Global Embedding =================\n",
        f"REPO_DIR: {REPO_DIR}\n",
        f"IMAGE_PATH: {image_path}\n",
        f"HUB_ENTRY: {hub_entry}\n",
        f"CKPT_PATH: {ckpt}\n",
        f"device: {device}\n",
        f"Test Global embedding DINOv3 numpy array -> {npy_path}\n",
        f"Test Global embedding DINOv3 csv row     -> {csv_path}\n",
        f"Test Global patch grid numpy array       -> {grid_path}\n",
        "\n================= Debug: Test Global Embedding =================\n",
    )

    print("Loading model and checkpoint...\n")
    model, _ = progress_bar(load_model, REPO_DIR, hub_entry, ckpt, device)

    print("Model and checkpoint loaded.\n")

    img_tensor = progress_bar(load_image_tensor, image_path.as_posix())
    print("Image loaded.\n")

    patch = model.patch_embed.patch_size
    patch_multiple = math.floor(target_res / patch[0])
    print(
        f"Model patch size: {patch}\n"
        f"Image resized to: {patch_multiple * patch[0]}x{patch_multiple * patch[1]}\n"
    )

    transform = progress_bar(
        build_transform,
        patch_size=patch[0],
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=True,
    )
    print(f"transform built: {transform}\n")

    input_tensor = progress_bar(transform, img_tensor).unsqueeze(0)
    print("Input tensor prepared:", input_tensor.shape, "\n")

    print("Extracting global feature and patch tokens...\n")
    with torch.inference_mode():
        global_vec = progress_bar(extract_global_feature, model, input_tensor, device)
        patch_tokens = progress_bar(extract_patch_tokens, model, input_tensor, device)

    print("================= Feature extraction completed =================\n")
    global_vec = global_vec.detach().cpu()

    patch_grid = None
    if patch_tokens is not None:
        patch_tokens = patch_tokens.detach().cpu()
        try:
            patch_grid = reshape_patch_tokens_to_grid(patch_tokens)
        except ValueError as err:
            print(f"[warn] patch grid reshape failed: {err}")
    else:
        print("[warn] patch tokens could not be extracted.")

    print("================= Saving features =================\n")
    print("Global feature shape:", tuple(global_vec.shape))
    print("Global feature:", global_vec.tolist())

    if patch_tokens is not None:
        print("Patch tokens shape:", tuple(patch_tokens.shape))
        if patch_grid is not None:
            print("Patch grid shape:", tuple(patch_grid.shape))

    print("\n================= Exporting features =================\n")
    global_arr = global_vec.numpy()
    progress_bar(np.save, npy_path, global_arr)
    progress_bar(np.savetxt, csv_path, global_arr[None, :], delimiter=",")

    print("\n================= Feature export completed =================\n")
    print(f"[saved] Test Global embedding DINOv3 numpy array -> {npy_path}")
    print(f"[saved] Test Global embedding DINOv3 csv row     -> {csv_path}")
    if patch_grid is not None:
        np.save(grid_path, patch_grid.numpy())
        print(f"[saved] Test Global patch grid numpy array   -> {grid_path}")


def main() -> None:
    run_test_global_embedding(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
        target_res=varTargetRes,
    )


if __name__ == "__main__":
    main()
