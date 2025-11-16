# project/Test_Embedding.py
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
    EMBED_ROOT,
    weights_path,
    file_prefix,
    img_path,
    load_image,
    sanitize_group_token,
    normalize_group_value,
)
from imatch.utils import (
    progress_bar,
    token_preview
)
from imatch.postprocess import process_patch_tokens

# 디버깅용, 단일 실행 파라미터
varAltitude = 450
varIndex = 1
varWeight = "vitb16"
varTargetRes = 1024

## 전역 상수
REPO_DIR = Path("/workspace/dinov3")
TOKEN_OUTPUT_KEYS = ("global", "patch", "grid")

## 출력 계획 정규화 함수: 
def _normalize_output_plan(
    plan: Optional[Dict[str, Dict[str, bool]]]
) -> Dict[str, Dict[str, bool]]:
    """
    정규화된 출력 계획을 반환합니다. plan이 None인 경우 모든 토큰 유형에 대해 npy 및 json 출력을 활성화합니다.
    각 토큰 유형에 대해 plan이 제공된 경우, npy 및 json 출력 여부를 개별적으로 설정합니다.
    입력:
        - plan: Optional[Dict[str, Dict[str, bool]]]
    출력:
        - Dict[str, Dict[str, bool]]

    e.g.1)
    _normalize_output_plan({
        "global": {"npy": True, "json": False},
        "patch": {"npy": True, "json": True},
        "grid": {"npy": False, "json": True}
    })
    → returns: {
        "global": {"npy": True, "json": False}, 
        "patch": {"npy": True, "json": True},
        "grid": {"npy": False, "json": True}
    }

    e.g.2)
    _normalize_output_plan(None)
    → returns: {
        "global": {"npy": True, "json": True}, 
        "patch": {"npy": True, "json": True},
        "grid": {"npy": True, "json": True}
    }
    """
    
    if plan is None:
        return {key: {"npy": True, "json": True} for key in TOKEN_OUTPUT_KEYS}
    normalized = {key: {"npy": False, "json": False} for key in TOKEN_OUTPUT_KEYS}
    for key in TOKEN_OUTPUT_KEYS:
        entry = plan.get(key) if isinstance(plan, dict) else None
        if isinstance(entry, dict):
            normalized[key]["npy"] = bool(entry.get("npy"))
            normalized[key]["json"] = bool(entry.get("json"))
    return normalized


def _build_context(
    altitude: int | str,
    index: int,
    weight: str,
    dataset_key: str | None,
    target_res: int,
    variant: str,
    embedding_cfg: Optional[str],
    variant_params: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Assemble frequently reused values for a single inference run."""
    hub_entry, key, dt_type = weights_path(weight) # e.g. "dinov3_vit7b16", "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth", "SAT"
    image_spec = img_path(altitude, index, dataset_key=dataset_key)
    prefix = file_prefix(image_spec.label, index) # e.g. "200_0150"

    # Derive embedding configuration when not explicitly provided.
    default_embedding_cfg = f"res{target_res}_ImageNet"
    resolved_embedding_cfg = embedding_cfg or default_embedding_cfg

    # Token naming follows: token_type → embedding_cfg → variant → weight_id → dataset_type → altitude → index
    label_display = normalize_group_value(altitude)
    label_token = sanitize_group_token(altitude)
    index_str = f"{int(index):04d}"
    global_base = f"GlobalToken_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dt_type}_{label_token}_{index_str}"
    patch_base = f"PatchToken_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dt_type}_{label_token}_{index_str}"
    patch_grid_base = f"PatchGrid_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dt_type}_{label_token}_{index_str}"

    export_root = EMBED_ROOT / weight
    altitude_dir = export_root / label_token
    global_dir = altitude_dir / "GlobalToken"
    patch_dir = altitude_dir / "PatchToken"
    grid_dir = altitude_dir / "PatchGrid"

    return {
        "hub_entry": hub_entry, # e.g. "dinov3_vit7b16"
        "key_path": key, # e.g. "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        "dataset_type": dt_type,
        "image_path": image_spec.path, # resolved absolute path
        "dataset_key": image_spec.dataset_key,
        "label_display": label_display,
        "label_token": label_token,
        "file_name": global_base,
        "patch_name": patch_base,
        "grid_name": patch_grid_base,
        "export_root": export_root,
        "altitude_dir": altitude_dir,
        "global_dir": global_dir,
        "patch_dir": patch_dir,
        "grid_dir": grid_dir,
        "embedding_cfg": resolved_embedding_cfg,
        "variant": variant,
        "variant_params": dict(variant_params or {}),
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
            "\033[91mAttributeError: Unable to infer patch size: model lacks patch_embed/patch_size/stem/downsample_layers metadata.\033[0m"
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
            raise TypeError(f"\033[91mTypeError: Unsupported patch size type = {type(patch)}\033[0m") from err

    if len(patch) == 1:
        patch = (patch[0], patch[0])
    elif len(patch) < 2:
        raise ValueError(f"\033[91mValueError: Resolved patch size has insufficient dimensions = {patch}\033[0m")

    return patch[0], patch[1]


def run_global_embedding(
    altitude: int | str,
    index: int,
    weight: str,
    target_res: int = 1024,
    variant: str = "raw",
    embedding_cfg: Optional[str] = None,
    variant_params: Optional[Dict[str, object]] = None,
    output_plan: Optional[Dict[str, Dict[str, bool]]] = None,
    dataset_key: Optional[str] = None,
) -> None:
    """Execute the full embedding pipeline for the given parameters."""
    ctx = _build_context(altitude, index, weight, dataset_key, target_res, variant, embedding_cfg, variant_params)
    hub_entry = ctx["hub_entry"]
    weight_path = ctx["key_path"]
    dataset_type = ctx["dataset_type"]
    image_path = ctx["image_path"]
    file_name = ctx["file_name"]
    patch_name = ctx["patch_name"]
    grid_name = ctx["grid_name"]
    global_dir = ctx["global_dir"]
    patch_dir = ctx["patch_dir"]
    grid_dir = ctx["grid_dir"]
    resolved_embedding_cfg = ctx["embedding_cfg"]
    resolved_variant = ctx["variant"]
    resolved_variant_params = dict(ctx["variant_params"])
    label_display = ctx["label_display"]
    index_str = ctx["index_str"]
    prefix = ctx["prefix"]

    plan = _normalize_output_plan(output_plan)
    global_plan = plan["global"]
    patch_plan = plan["patch"]
    grid_plan = plan["grid"]

    emit_global = bool(global_plan["npy"] or global_plan["json"])
    emit_patch = bool(patch_plan["npy"] or patch_plan["json"])
    emit_grid = bool(grid_plan["npy"] or grid_plan["json"])
    need_global = emit_global
    need_patch = emit_patch or emit_grid

    npy_path = global_dir / f"{file_name}.npy"
    patch_path = patch_dir / f"{patch_name}.npy"
    grid_path = grid_dir / f"{grid_name}.npy"
    global_meta_path = global_dir / f"{file_name}_meta.json"
    patch_meta_path = patch_dir / f"{patch_name}_meta.json"
    grid_meta_path = grid_dir / f"{grid_name}_meta.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        "\n",
        "================= Debug: Embedding (Datasets) =================\n",
        "INPUT: \n",
        f"\tREPO_DIR: \033[33m{REPO_DIR}\033[0m\n", # e.g. /workspace/dinov3
        f"\tIMAGE_PATH: \033[33m{image_path}\033[0m\n", # e.g. /opt/datasets/250912161658_200/250912161658_200_0150.jpg
        f"\tHUB_ENTRY: \033[33m{hub_entry}\033[0m\n", # e.g. dinov3_vits16
        f"\tweight_path: \033[33m{weight_path}\033[0m\n", # e.g. /opt/weights/03_ViT_SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
        f"\tdevice: \033[33m{device}\033[0m\n", # e.g. "cuda"
        "OUTPUT: \n",
        f"\t[config] embedding_cfg: \033[33m{resolved_embedding_cfg}\033[0m\n",
        f"\t[config] variant: \033[33m{resolved_variant}\033[0m\n",
        f"\t[config] group/index: \033[33m{label_display}/{index_str}\033[0m (prefix=\033[33m{prefix}\033[0m)\n",
        f"\t[outputs] Test Global embedding DINOv3 numpy array -> \033[34m{npy_path}\033[0m\n",
        f"\t[outputs] Test Patch token numpy array             -> \033[34m{patch_path}\033[0m\n",
        f"\t[outputs] Test Patch grid numpy array              -> \033[34m{grid_path}\033[0m\n",
        "================= Debug: Embedding (Datasets) =================\n",
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
    global_tokens: Optional[torch.Tensor] = None
    patch_tokens: Optional[torch.Tensor] = None
    with torch.inference_mode():
        if need_global:
            g_start = time.perf_counter()
            global_tokens = progress_bar(global_embedding, model, input_tensor, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["global_forward"] = (time.perf_counter() - g_start) * 1000.0

        if need_patch:
            p_start = time.perf_counter()
            patch_tokens = progress_bar(patch_embedding, model, input_tensor, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["patch_forward"] = (time.perf_counter() - p_start) * 1000.0
    
    patch_grid = None
    patch_numpy = None
    patch_post_info = None
    if global_tokens is not None:
        global_tokens = global_tokens.detach().cpu()
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
        if emit_patch:
            patch_numpy = patch_tokens.numpy()
        grid_from_info = None
        if patch_post_info is not None and "grid" in patch_post_info:
            grid_from_info = patch_post_info.pop("grid")
        if patch_post_info is not None and "grid_shape" in patch_post_info:
            patch_post_info.pop("grid_shape")
        if emit_grid:
            if grid_from_info is not None:
                patch_grid = grid_from_info.detach().cpu() if isinstance(grid_from_info, torch.Tensor) else torch.as_tensor(grid_from_info)
            else:
                try:
                    patch_grid = patch2grid(patch_tokens)
                except ValueError as err:
                    print(f"\t\033[91m[WARN 1: Global Embedding]  patch grid reshape failed: {err}\033[0m")
    elif need_patch:
        print("\t\033[91m[WARN 2: Global Embedding] patch tokens could not be extracted.\033[0m")

    if global_tokens is not None:
        print("\t[Global Embedding 6] Global feature shape:", tuple(global_tokens.shape))
        print("\t[Global Embedding 7] Global feature:", token_preview(global_tokens))
    elif need_global:
        print("\t\033[91m[WARN: Global Embedding] Global tokens were requested but not produced.\033[0m")

    if patch_tokens is not None:
        print("\t[Global Embedding 8] Patch tokens shape:", tuple(patch_tokens.shape))
        if patch_post_info is not None:
            kept = patch_post_info.get("kept_tokens", patch_tokens.shape[0])
            keep_ratio = patch_post_info.get("keep_ratio", 1.0)
            print(f"\t[Global Embedding 8A] Patch variant '{resolved_variant}' kept {kept} tokens ({keep_ratio:.3f} ratio)")
        if patch_grid is not None:
            print("\t[Global Embedding 9] Patch grid shape:", tuple(patch_grid.shape))
            print("\t[Global Embedding 10] Patch grid preview:", token_preview(patch_grid))
    elif need_patch:
        print("\t\033[91m[WARN: Global Embedding] Patch tokens were requested but not produced.\033[0m")
    print(">>>>>>>>>>>>>>> Extracting global and patch tokens completed\n")

    print("\nFeature exporting")
    print(f"<<< Test Global Embedding OUTPUT >>>\n")

    global_array = None
    if global_plan["npy"]:
        if global_tokens is not None:
            global_array = global_tokens.numpy()
            global_dir.mkdir(parents=True, exist_ok=True)
            progress_bar(np.save, npy_path, global_array)
            print(f"\t\033[32m[saved] Test Global embedding DINOv3 numpy array -> {npy_path}\033[0m")
        else:
            print("\t\033[91m[warn] Global npy requested but tokens unavailable.\033[0m")
    else:
        print("\t\033[91m[skip] Global npy disabled by configuration.\033[0m")
    if patch_plan["npy"]:
        if patch_numpy is not None:
            patch_dir.mkdir(parents=True, exist_ok=True)
            progress_bar(np.save, patch_path, patch_numpy)
            print(f"\t\033[32m[saved] Test Patch token numpy array       -> {patch_path}\033[0m")
        else:
            print("\t\033[91m[warn] Patch npy requested but tokens unavailable.\033[0m")
    else:
        print("\t\033[91m[skip] Patch npy disabled by configuration.\033[0m")
    grid_array = None
    if emit_grid and patch_grid is not None:
        if isinstance(patch_grid, torch.Tensor):
            grid_array = patch_grid.detach().cpu().numpy()
        else:
            grid_array = np.asarray(patch_grid)

    if grid_plan["npy"]:
        if grid_array is not None:
            grid_dir.mkdir(parents=True, exist_ok=True)
            np.save(grid_path, grid_array)
            print(f"\t\033[32m[saved] Test Patch grid numpy array         -> {grid_path}\033[0m")
        else:
            print("\t\033[91m[warn] Patch grid npy requested but grid unavailable.\033[0m")
    else:
        print("\t\033[91m[skip] Patch grid npy disabled by configuration.\033[0m")

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
        "vector": _file_entry(npy_path) if global_plan["npy"] and global_array is not None else None,
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    patch_files = {
        "vector": None,
        "patch_tokens": _file_entry(patch_path) if patch_plan["npy"] and patch_numpy is not None else None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    grid_files = {
        "vector": _file_entry(grid_path) if grid_plan["npy"] and grid_array is not None else None,
        "patch_tokens": None,
        "patch_grid": None,
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

    if global_plan["json"] and global_tokens is not None:
        global_dir.mkdir(parents=True, exist_ok=True)
        global_meta = {
            "run_id": file_name,
            "token_type": "GlobalToken",
            "config": dict(common_config),
            "files": global_files,
            "metrics": {
                "token_count": 1,
                "embedding_dim": int(global_tokens.shape[0]) if global_tokens is not None else None,
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
    elif global_plan["json"]:
        print("\t\033[91m[warn] Global meta requested but tokens unavailable.\033[0m")

    if patch_plan["json"] and patch_tokens is not None:
        patch_dir.mkdir(parents=True, exist_ok=True)
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
    elif patch_plan["json"]:
        print("\t\033[91m[warn] Patch meta requested but tokens unavailable.\033[0m")

    if grid_plan["json"] and grid_array is not None:
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_h = int(grid_array.shape[0]) if grid_array.ndim >= 1 else None
        grid_w = int(grid_array.shape[1]) if grid_array.ndim >= 2 else None
        grid_dim = int(grid_array.shape[2]) if grid_array.ndim >= 3 else None
        grid_metrics = {
            "token_count": int(grid_h * grid_w) if grid_h is not None and grid_w is not None else None,
            "grid_shape": [grid_h, grid_w] if grid_h is not None and grid_w is not None else None,
            "embedding_dim": grid_dim,
            "matching_count": None,
            "mutual_knn_tokens": None,
            "keep_ratio": patch_post_info.get("keep_ratio") if patch_post_info else None,
            "recall@1": None,
            "recall@5": None,
            "recall@10": None,
            "mAP": None,
            "top1_precision": None,
        }

        grid_meta = {
            "run_id": grid_name,
            "token_type": "PatchGrid",
            "config": dict(common_config),
            "files": grid_files,
            "metrics": grid_metrics,
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak_mem_mb,
                "embedding_storage_bytes": _sum_sizes(grid_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(grid_meta_path, grid_meta)
    elif grid_plan["json"]:
        print("\t\033[91m[warn] Patch grid meta requested but grid unavailable.\033[0m")


def main() -> None:
    run_global_embedding(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
        target_res=varTargetRes,
    )


if __name__ == "__main__":
    main()
