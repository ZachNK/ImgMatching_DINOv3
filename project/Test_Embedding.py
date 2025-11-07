from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
import numpy as np
import torch
from imatch.pretrained import pretrained_model
from imatch.preprocess import build_transform
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
from imatch.postprocess import process_patch_tokens

# Default parameters kept for manual single-run usage.
varAltitude = 450
varIndex = 1
varWeight = "vitb16"
varTargetRes = 1024

REPO_DIR = Path("/workspace/dinov3")


def _build_context(
    altitude: int,
    index: int,
    weight: str,
    target_res: int,
    variant: str,
    embedding_cfg: Optional[str],
    variant_params: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Assemble frequently reused values for a single inference run."""
    hub_entry, key, dt_type = weights_path(weight) # e.g. "dinov3_vit7b16", "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth", "SAT"
    img_dir_a, img_dir_b = img_path(altitude, index) # e.g. "250912161658_200", "250912161658_200_0150"
    prefix = file_prefix(altitude, index) # e.g. "200_0150"

    # Derive embedding configuration when not explicitly provided.
    resolved_embedding_cfg = embedding_cfg or f"res{target_res}_ImageNet{dt_type}"

    # Token naming follows: token_type → embedding_cfg → variant → weight_id → dataset_type → altitude → index
    altitude_str = f"{int(altitude)}"
    index_str = f"{int(index):04d}"
    global_base = f"GlobalToken_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dt_type}_{altitude_str}_{index_str}"
    patch_base = f"PatchToken_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dt_type}_{altitude_str}_{index_str}"
    patch_grid_base = f"PatchGrid_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dt_type}_{altitude_str}_{index_str}"

    return {
        "hub_entry": hub_entry, # e.g. "dinov3_vit7b16"
        "key_path": key, # e.g. "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        "dataset_type": dt_type,
        "image_path": DATASET_ROOT / f"{img_dir_a}/{img_dir_b}.jpg", # e.g. "/opt/datasets/250912161658_200/250912161658_200_0150.jpg"
        "file_name": global_base,
        "patch_name": patch_base,
        "grid_name": patch_grid_base,
        "export_dir": EMBED_ROOT / weight, # e.g. /exports/dinov3_embeds/vitb16
        "embedding_cfg": resolved_embedding_cfg,
        "variant": variant,
        "variant_params": dict(variant_params or {}),
        "altitude_str": altitude_str,
        "index_str": index_str,
        "prefix": prefix,
    }


def _file_entry(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": path.name,
        "size_bytes": stat.st_size,
    }


def _write_meta(meta_path: Path, payload: Dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _gather_gpu_stats(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


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
    variant: str = "raw",
    embedding_cfg: Optional[str] = None,
    variant_params: Optional[Dict[str, object]] = None,
) -> None:
    """Execute the full embedding pipeline for the given parameters."""
    ctx = _build_context(altitude, index, weight, target_res, variant, embedding_cfg, variant_params)
    hub_entry = ctx["hub_entry"]
    weight_path = ctx["key_path"]
    dataset_type = ctx["dataset_type"]
    image_path = ctx["image_path"]
    file_name = ctx["file_name"]
    patch_name = ctx["patch_name"]
    grid_name = ctx["grid_name"]
    export_dir = ctx["export_dir"]
    resolved_embedding_cfg = ctx["embedding_cfg"]
    resolved_variant = ctx["variant"]
    resolved_variant_params = dict(ctx["variant_params"])
    altitude_str = ctx["altitude_str"]
    index_str = ctx["index_str"]
    prefix = ctx["prefix"]

    export_dir.mkdir(parents=True, exist_ok=True)
    npy_path = export_dir / f"{file_name}.npy"
    patch_path = export_dir / f"{patch_name}.npy"
    grid_path = export_dir / f"{grid_name}.npy"
    global_meta_path = export_dir / f"{file_name}_meta.json"
    patch_meta_path = export_dir / f"{patch_name}_meta.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        "\n",
        "================= Debug: Test Global Embedding =================\n",
        "INPUT: \n",
        f"\tREPO_DIR: {REPO_DIR}\n", # e.g. /workspace/dinov3
        f"\tIMAGE_PATH: {image_path}\n", # e.g. /opt/datasets/250912161658_200/250912161658_200_0150.jpg
        f"\tHUB_ENTRY: {hub_entry}\n", # e.g. dinov3_vits16
        f"\tkey_path: {weight_path}\n", # e.g. /opt/weights/03_ViT_SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
        f"\tdevice: {device}\n", # e.g. "cuda"
        "OUTPUT: \n",
        f"\t[config] embedding_cfg: {resolved_embedding_cfg}\n",
        f"\t[config] variant: {resolved_variant}\n",
        f"\t[config] altitude/index: {altitude_str}/{index_str} (prefix={prefix})\n",
        f"\tTest Global embedding DINOv3 numpy array -> {npy_path}\n",
        f"\tTest Patch token numpy array             -> {patch_path}\n",
        f"\tTest Patch grid numpy array              -> {grid_path}\n",
        "================= Debug: Test Global Embedding =================\n",
    )

    print("\nLoading model and weight")
    model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
    print(">>>>>>>>>>>>>>> Loading model and weight completed\n")

    print("\nPreparing input image")
    img_tensor = progress_bar(load_image, image_path.as_posix())
    print(f"\t[Global Embedding 1] Input image shape: {img_tensor.shape}")
    print(f">>>>>>>>>>>>>>> Preparing input image completed\n")

    print("\nResizing and transforming input")
    patch_h, patch_w = _resolve_patch_size(model)
    patch_multiple = max(1, math.floor(target_res / patch_h))
    print(
        f"\t[Global Embedding 2] Model patch size: {(patch_h, patch_w)}\n"
        f"\t[Global Embedding 3] Image resized to: {patch_multiple * patch_h}x{patch_multiple * patch_w}\n"
    )

    transform = progress_bar(
        build_transform,
        patch_size=patch_h,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=dataset_type,
    )
    print(f"\t[Global Embedding 4]\n transform: {transform}")
    print(f">>>>>>>>>>>>>>> Resizing and transforming input completed\n")

    print("\nPreparing input tensor")
    input_tensor = progress_bar(transform, img_tensor).unsqueeze(0)
    print(f"\t[Global Embedding 5] Input tensor shape: {input_tensor.shape}")
    print(f">>>>>>>>>>>>>>> Preparing input tensor completed\n")

    print("\nExtracting global and patch tokens")
    timings: Dict[str, Optional[float]] = {
        "global_forward": None,
        "patch_forward": None,
        "postprocess": None,
        "index_build": None,
        "query": None,
        "pipeline_total": None,
    }
    pipeline_start = time.perf_counter()
    with torch.inference_mode():
        g_start = time.perf_counter()
        global_tokens = progress_bar(global_embedding, model, input_tensor, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["global_forward"] = (time.perf_counter() - g_start) * 1000.0

        p_start = time.perf_counter()
        patch_tokens = progress_bar(patch_embedding, model, input_tensor, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["patch_forward"] = (time.perf_counter() - p_start) * 1000.0
    
    global_tokens = global_tokens.detach().cpu()
    patch_grid = None
    patch_numpy = None
    patch_post_info = None
    if patch_tokens is not None:
        patch_tokens = patch_tokens.detach().cpu()
        post_start = time.perf_counter()
        processed_tokens, patch_post_info = process_patch_tokens(
            patch_tokens,
            resolved_variant,
            resolved_variant_params,
        )
        timings["postprocess"] = (time.perf_counter() - post_start) * 1000.0
        patch_tokens = processed_tokens
        patch_numpy = patch_tokens.numpy()
        grid_from_info = None
        if patch_post_info is not None and "grid" in patch_post_info:
            grid_from_info = patch_post_info.pop("grid")
        if patch_post_info is not None and "grid_shape" in patch_post_info:
            patch_post_info.pop("grid_shape")
        if grid_from_info is not None:
            patch_grid = grid_from_info.detach().cpu() if isinstance(grid_from_info, torch.Tensor) else torch.as_tensor(grid_from_info)
        else:
            try:
                patch_grid = patch2grid(patch_tokens)
            except ValueError as err:
                print(f"\t[WARN 1: Global Embedding]  patch grid reshape failed: {err}")
    else:
        print("\t[WARN 2: Global Embedding] patch tokens could not be extracted.")

    print("\t[Global Embedding 6] Global feature shape:", tuple(global_tokens.shape))
    print("\t[Global Embedding 7] Global feature:", token_preview(global_tokens))

    if patch_tokens is not None:
        print("\t[Global Embedding 8] Patch tokens shape:", tuple(patch_tokens.shape))
        if patch_post_info is not None:
            kept = patch_post_info.get("kept_tokens", patch_tokens.shape[0])
            keep_ratio = patch_post_info.get("keep_ratio", 1.0)
            print(f"\t[Global Embedding 8A] Patch variant '{resolved_variant}' kept {kept} tokens ({keep_ratio:.3f} ratio)")
        if patch_grid is not None:
            print("\t[Global Embedding 9] Patch grid shape:", tuple(patch_grid.shape))
            print("\t[Global Embedding 10] Patch grid preview:", token_preview(patch_grid))
    print(">>>>>>>>>>>>>>> Extracting global and patch tokens completed\n")

    print("\nFeature exporting")
    global_arr = global_tokens.numpy()
    progress_bar(np.save, npy_path, global_arr)
    print(f"<<< Test Global Embedding OUTPUT >>>\n")
    print(f"\t[saved] Test Global embedding DINOv3 numpy array -> {npy_path}") # e.g. /exports/dinov3_embeds/vit7b16sat/Global_dinov3_vit7b16_SAT_200_0150.npy
    if patch_numpy is not None:
        progress_bar(np.save, patch_path, patch_numpy)
        print(f"\t[saved] Test Patch token numpy array       -> {patch_path}")
    if patch_grid is not None:
        if isinstance(patch_grid, torch.Tensor):
            grid_array = patch_grid.detach().cpu().numpy()
        else:
            grid_array = np.asarray(patch_grid)
        np.save(grid_path, grid_array)
        print(f"\t[saved] Test Patch grid numpy array         -> {grid_path}") # e.g. /exports/dinov3_embeds/vit7b16sat/PatchGrid_res1024_ImageNetSAT_raw_...
    print(">>>>>>>>>>>>>>> Feature exporting completed\n")

    timings["pipeline_total"] = (time.perf_counter() - pipeline_start) * 1000.0
    gpu_peak_mem_mb = _gather_gpu_stats(device)

    def _sum_sizes(entries: Dict[str, Optional[Dict[str, Any]]]) -> Optional[int]:
        total = 0
        has_file = False
        for entry in entries.values():
            if entry and "size_bytes" in entry:
                total += int(entry["size_bytes"])
                has_file = True
        return total if has_file else None

    global_files = {
        "vector": _file_entry(npy_path),
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    patch_files = {
        "vector": None,
        "patch_tokens": _file_entry(patch_path) if patch_numpy is not None else None,
        "patch_grid": _file_entry(grid_path) if patch_grid is not None else None,
        "dense_vis": None,
        "index": None,
    }

    used_variant_params = {}
    if patch_post_info and "params" in patch_post_info:
        params = patch_post_info["params"]
        if isinstance(params, dict):
            used_variant_params = dict(params)
    if not used_variant_params:
        used_variant_params = dict(resolved_variant_params)

    rotations_config = used_variant_params.get("rotations") if isinstance(used_variant_params, dict) else None
    aggregation_config = used_variant_params.get("aggregation") if isinstance(used_variant_params, dict) else None

    common_config = {
        "embedding_cfg": resolved_embedding_cfg,
        "variant": resolved_variant,
        "variant_params": used_variant_params,
        "weight_id": hub_entry,
        "dataset_type": dataset_type,
        "altitude": altitude,
        "index": index,
        "prefix": prefix,
        "target_res": target_res,
        "rotations": rotations_config if isinstance(rotations_config, (list, tuple)) else [0],
        "aggregation": aggregation_config if isinstance(aggregation_config, str) else "single",
    }

    global_meta = {
        "run_id": file_name,
        "token_type": "GlobalToken",
        "config": dict(common_config),
        "files": global_files,
        "metrics": {
            "token_count": 1,
            "embedding_dim": int(global_arr.shape[0]),
            "matching_count": None,
            "mutual_knn_tokens": None,
            "keep_ratio": None,
            "recall@1": None,
            "recall@5": None,
            "recall@10": None,
            "mAP": None,
            "top1_precision": None,
        },
        "timing_ms": dict(timings),
        "resources": {
            "gpu_peak_mem_mb": gpu_peak_mem_mb,
            "embedding_storage_bytes": _sum_sizes(global_files),
            "index_size_bytes": None,
        },
    }
    _write_meta(global_meta_path, global_meta)

    if patch_numpy is not None:
        patch_metrics = {
            "token_count": int(patch_tokens.shape[0]),
            "embedding_dim": int(patch_tokens.shape[1]) if patch_tokens.ndim == 2 else None,
            "matching_count": patch_post_info.get("kept_tokens") if patch_post_info else int(patch_tokens.shape[0]),
            "mutual_knn_tokens": patch_post_info.get("kept_tokens") if patch_post_info and resolved_variant == "mutual" else None,
            "keep_ratio": patch_post_info.get("keep_ratio") if patch_post_info else 1.0,
            "recall@1": None,
            "recall@5": None,
            "recall@10": None,
            "mAP": None,
            "top1_precision": None,
        }

        patch_meta = {
            "run_id": patch_name,
            "token_type": "PatchToken",
            "config": dict(common_config),
            "files": patch_files,
            "metrics": patch_metrics,
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak_mem_mb,
                "embedding_storage_bytes": _sum_sizes(patch_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(patch_meta_path, patch_meta)


def main() -> None:
    run_global_embedding(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
        target_res=varTargetRes,
    )


if __name__ == "__main__":
    main()
