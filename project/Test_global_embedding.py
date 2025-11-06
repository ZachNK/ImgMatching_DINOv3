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
from imatch.pretrained import pretrained_model
from imatch.imageprocessing import build_transform
from imatch.extracting import (
    global_embedding,
    patch_embedding,
    patch2grid
)
from imatch.loading import (
    DATASET_ROOT,
    EMBED_ROOT,
    weights_path,
    file_prefix,
    img_path,
    load_image
)
from imatch.utils import (
    progress_bar,
    token_preview
)

# Default parameters kept for manual single-run usage.
varAltitude = 400
varIndex = 160
varWeight = "cxLarge"
varTargetRes = 1024

REPO_DIR = Path("/workspace/dinov3")
_EXPORT_SUBDIR = varWeight # e.g. vit7b16sat


def _build_context(altitude: int, index: int, weight: str) -> Dict[str, object]:
    """Assemble frequently reused values for a single inference run."""
    hub_entry, key, dt_type = weights_path(weight) # e.g. "dinov3_vit7b16", "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth", "SAT"
    img_dir_a, img_dir_b = img_path(altitude, index) # e.g. "250912161658_200", "250912161658_200_0150"
    prefix = file_prefix(altitude, index) # e.g. "200_0150"

    return {
        "hub_entry": hub_entry, # e.g. "dinov3_vit7b16"
        "key_path": key, # e.g. "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        "dataset_type": dt_type,
        "image_path": DATASET_ROOT / f"{img_dir_a}/{img_dir_b}.jpg", # e.g. "/opt/datasets/250912161658_200/250912161658_200_0150.jpg"
        "file_name": f"Global_{hub_entry}_{dt_type}_{prefix}", # e.g. "Global_dinov3_vit7b16_SAT_200_0150"
        "grid_name": f"Global_grid_{hub_entry}_{dt_type}_{prefix}", # e.g. "Global_grid_dinov3_vit7b16_SAT_200_0150"
        "export_dir": EMBED_ROOT / _EXPORT_SUBDIR, # e.g. "/exports/dinov3_debug/Test_global_embedding+dense_feature/1106" --> /exports/dinov3_embeds/vit7b16sat
    }


def _resolve_patch_size(model: torch.nn.Module) -> tuple[int, int]:
    """
    DINOv3 ConvNeXt checkpoints do not expose `patch_embed`, so gracefully try
    the common variants and always return a 2D patch size tuple.
    """
    patch = None

    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "patch_size"):
        patch = patch_embed.patch_size

    if patch is None and hasattr(model, "patch_size"):
        patch = model.patch_size

    if patch is None and hasattr(model, "stem"):
        stem = getattr(model.stem, "0", model.stem[0] if len(model.stem) > 0 else None)
        if stem is not None and hasattr(stem, "kernel_size"):
            patch = stem.kernel_size

    if patch is None and hasattr(model, "downsample_layers"):
        first = model.downsample_layers[0] if len(model.downsample_layers) > 0 else None
        if first is not None:
            conv = getattr(first, "0", first[0] if len(first) > 0 else None)
            if conv is not None and hasattr(conv, "kernel_size"):
                patch = conv.kernel_size

    if patch is None:
        raise AttributeError(
            "AttributeError: Unable to infer patch size: model lacks patch_embed/patch_size/stem/downsample_layers metadata."
        )

    if isinstance(patch, torch.Size):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, (list, tuple)):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, int):
        patch = (patch, patch)
    else:
        # Fall back to trying to iterate, otherwise bail.
        try:
            patch = tuple(int(p) for p in patch)  # type: ignore[arg-type]
        except TypeError as err:  # pragma: no cover - defensive
            raise TypeError(f"TypeError: Unsupported patch size type = {type(patch)}") from err

    if len(patch) == 1:
        patch = (patch[0], patch[0])
    elif len(patch) < 2:
        raise ValueError(f"ValueError: Resolved patch size has insufficient dimensions = {patch}")

    return patch[0], patch[1]


def run_global_embedding(
    altitude: int,
    index: int,
    weight: str,
    target_res: int = 1024,
) -> None:
    """Execute the full embedding pipeline for the given parameters."""
    ctx = _build_context(altitude, index, weight)
    hub_entry = ctx["hub_entry"]
    weight_path = ctx["key_path"]
    dataset_type = ctx["dataset_type"]
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
        "\n",
        "================= Debug: Test Global Embedding =================\n",
        "INPUT: \n",
        f"  REPO_DIR: {REPO_DIR}\n", # e.g. /workspace/dinov3
        f"  IMAGE_PATH: {image_path}\n", # e.g. /opt/datasets/250912161658_200/250912161658_200_0150.jpg
        f"  HUB_ENTRY: {hub_entry}\n", # e.g. dinov3_vits16
        f"  key_path: {weight_path}\n", # e.g. /opt/weights/03_ViT_SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
        f"  device: {device}\n", # e.g. "cuda"
        "OUTPUT: \n",
        f"  Test Global embedding DINOv3 numpy array -> {npy_path}\n", # e.g. /exports/dinov3_embeds/vit7b16sat/Global_dinov3_vit7b16_SAT_200_0150.npy
        f"  Test Global embedding DINOv3 csv row     -> {csv_path}\n", # e.g. /exports/dinov3_embeds/vit7b16sat/Global_dinov3_vit7b16_SAT_200_0150.csv
        f"  Test Global patch grid numpy array       -> {grid_path}\n", # e.g. /exports/dinov3_embeds/vit7b16sat/Global_grid_dinov3_vit7b16_SAT_200_0150.npy
        "================= Debug: Test Global Embedding =================\n",
    )

    print("\nLoading model and weight")
    model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
    print(">>>>>>>>>>>>>>> Loading model and weight completed\n")

    print("\nPreparing input image")
    img_tensor = progress_bar(load_image, image_path.as_posix())
    print(f"  [Global Embedding 1] Input image shape: {img_tensor.shape}")
    print(f">>>>>>>>>>>>>>> Preparing input image completed\n")

    print("\nResizing and transforming input")
    patch_h, patch_w = _resolve_patch_size(model)
    patch_multiple = max(1, math.floor(target_res / patch_h))
    print(
        f"  [Global Embedding 2] Model patch size: {(patch_h, patch_w)}\n"
        f"  [Global Embedding 3] Image resized to: {patch_multiple * patch_h}x{patch_multiple * patch_w}\n"
    )

    transform = progress_bar(
        build_transform,
        patch_size=patch_h,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=dataset_type,
    )
    print(f"  [Global Embedding 4]\n transform: {transform}")
    print(f">>>>>>>>>>>>>>> Resizing and transforming input completed\n")

    print("\nPreparing input tensor")
    input_tensor = progress_bar(transform, img_tensor).unsqueeze(0)
    print(f"  [Global Embedding 5] Input tensor shape: {input_tensor.shape}")
    print(f">>>>>>>>>>>>>>> Preparing input tensor completed\n")

    print("\nExtracting global and patch tokens")
    with torch.inference_mode():
        global_tokens = progress_bar(global_embedding, model, input_tensor, device)
        patch_tokens = progress_bar(patch_embedding, model, input_tensor, device)
    
    global_tokens = global_tokens.detach().cpu()
    patch_grid = None
    if patch_tokens is not None:
        patch_tokens = patch_tokens.detach().cpu()
        try:
            patch_grid = patch2grid(patch_tokens)
        except ValueError as err:
            print(f"  [WARN 1: Global Embedding]  patch grid reshape failed: {err}")
    else:
        print("  [WARN 2: Global Embedding] patch tokens could not be extracted.")

    print("  [Global Embedding 6] Global feature shape:", tuple(global_tokens.shape))
    print("  [Global Embedding 7] Global feature:", token_preview(global_tokens))

    if patch_tokens is not None:
        print("  [Global Embedding 8] Patch tokens shape:", tuple(patch_tokens.shape))
        if patch_grid is not None:
            print("  [Global Embedding 9] Patch grid shape:", tuple(patch_grid.shape))
            print("  [Global Embedding 10] Patch grid preview:", token_preview(patch_grid))
    print(">>>>>>>>>>>>>>> Extracting global and patch tokens completed\n")

    print("\nFeature exporting")
    global_arr = global_tokens.numpy()
    progress_bar(np.save, npy_path, global_arr)
    progress_bar(np.savetxt, csv_path, global_arr[None, :], delimiter=",")
    print(f"<<< Test Global Embedding OUTPUT >>>\n")
    print(f"  [saved] Test Global embedding DINOv3 numpy array -> {npy_path}") # e.g. /exports/dinov3_embeds/vit7b16sat/Global_dinov3_vit7b16_SAT_200_0150.npy
    print(f"  [saved] Test Global embedding DINOv3 csv row     -> {csv_path}") # e.g. /exports/dinov3_embeds/vit7b16sat/Global_dinov3_vit7b16_SAT_200_0150.csv
    if patch_grid is not None:
        np.save(grid_path, patch_grid.numpy())
        print(f"  [saved] Test Global patch grid numpy array   -> {grid_path}") # e.g. /exports/dinov3_embeds/vit7b16sat/Global_grid_dinov3_vit7b16_SAT_200_0150.npy
    print(">>>>>>>>>>>>>>> Feature exporting completed\n")


def main() -> None:
    run_global_embedding(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
        target_res=varTargetRes,
    )


if __name__ == "__main__":
    main()
