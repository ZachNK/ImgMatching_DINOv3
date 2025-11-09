"""
Extract embeddings for query images generated via rotations/crops.

This script mirrors Test_Embedding but reads images directly from the query
directories (e.g. /exports/Q2509...) and stores outputs under
/exports/dinov3_embed/Q{weight_key}/....
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from imatch.extracting import global_embedding, patch_embedding, patch2grid
from imatch.preprocess import build_transform
from imatch.loading import weights_path, load_image
from imatch.postprocess import process_patch_tokens
from imatch.pretrained import pretrained_model
from imatch.utils import progress_bar, token_preview


QUERY_DIRS: Sequence[Path] = (
    Path("/exports/Q250912150549_400"),
    Path("/exports/Q250912154506_300"),
    Path("/exports/Q250912161658_200"),
)

"""
    Path("/exports/Q250912150549_400"),
    Path("/exports/Q250912154506_300"),
    Path("/exports/Q250912161658_200"),
"""

VAR_WEIGHT_KEYS: Sequence[str] = ("vitb16", "vits16+")
VAR_TARGET_RES = 1024
VARIANT = "raw"
VARIANT_PARAMS: Dict[str, object] = {}

QUERY_EMBED_ROOT = Path("/exports/dinov3_query_embeds")
REPO_DIR = Path("/workspace/dinov3")

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class QueryInfo:
    scene: str
    altitude: int
    index: int
    tag: str
    identifier: str
    source: Path
    query_dir: Path


def _parse_query_filename(path: Path) -> QueryInfo:
    stem = path.stem  # e.g. 250912150549_400_0001_rot045_crop50
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"\033[91mUnexpected query filename format: {path.name}\033[0m")

    scene = parts[0]
    altitude = int(parts[1])
    index = int(parts[2])
    tag = "_".join(parts[3:])
    identifier = f"{scene}_{altitude}_{index:04d}_{tag}"
    return QueryInfo(
        scene=scene,
        altitude=altitude,
        index=index,
        tag=tag,
        identifier=identifier,
        source=path,
        query_dir=path.parent,
    )


def _file_entry(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    stat = path.stat()
    return {"path": path.name, "size_bytes": stat.st_size}


def _write_meta(meta_path: Path, payload: Dict[str, object]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _gather_gpu_stats(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _resolve_patch_size(model: torch.nn.Module) -> Tuple[int, int]:
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
        raise AttributeError("\033[91mUnable to infer patch size from model.\033[0m")

    if isinstance(patch, torch.Size):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, (list, tuple)):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, int):
        patch = (patch, patch)
    else:
        patch = tuple(int(p) for p in patch)  # type: ignore[arg-type]

    if len(patch) == 1:
        patch = (patch[0], patch[0])
    elif len(patch) < 2:
        raise ValueError(f"\033[91mResolved patch size invalid: {patch}\033[0m")

    return patch[0], patch[1]


def process_query_image(
    model: torch.nn.Module,
    device: torch.device,
    hub_entry: str,
    dataset_type: str,
    weight_key: str,
    info: QueryInfo,
    target_res: int,
    embedding_cfg: Optional[str],
    variant: str,
    variant_params: Dict[str, object],
) -> None:
    resolved_embedding_cfg = embedding_cfg or f"res{target_res}_ImageNet{dataset_type}"
    output_dir = QUERY_EMBED_ROOT / f"Q{weight_key}" / info.query_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    global_base = f"QueryGlobal_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dataset_type}_{info.identifier}"
    patch_base = f"QueryPatchToken_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dataset_type}_{info.identifier}"
    grid_base = f"QueryPatchGrid_{resolved_embedding_cfg}_{variant}_{hub_entry}_{dataset_type}_{info.identifier}"

    npy_path = output_dir / f"{global_base}.npy"
    csv_path = output_dir / f"{global_base}.csv"
    patch_path = output_dir / f"{patch_base}.npy"
    grid_path = output_dir / f"{grid_base}.npy"
    global_meta_path = output_dir / f"{global_base}_meta.json"
    patch_meta_path = output_dir / f"{patch_base}_meta.json"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"\n[Query] Processing {info.source} -> {output_dir}")

    img_tensor = progress_bar(load_image, info.source.as_posix())

    patch_h, patch_w = _resolve_patch_size(model)
    patch_multiple = max(1, math.floor(target_res / patch_h))

    transform = progress_bar(
        build_transform,
        patch_size=patch_h,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=dataset_type,
    )
    input_tensor = progress_bar(transform, img_tensor).unsqueeze(0)

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
            variant,
            variant_params,
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
                print(f"\033[91m[WARN] Query patch grid reshape failed: {err}\033[0m")
    else:
        print("\033[91m[WARN] Patch tokens could not be extracted.\033[0m")

    global_arr = global_tokens.numpy()
    progress_bar(np.save, npy_path, global_arr)
    progress_bar(np.savetxt, csv_path, global_arr[None, :], delimiter=",")
    if patch_numpy is not None:
        progress_bar(np.save, patch_path, patch_numpy)
    if patch_grid is not None:
        if isinstance(patch_grid, torch.Tensor):
            grid_array = patch_grid.detach().cpu().numpy()
        else:
            grid_array = np.asarray(patch_grid)
        np.save(grid_path, grid_array)

    timings["pipeline_total"] = (time.perf_counter() - pipeline_start) * 1000.0
    gpu_peak = _gather_gpu_stats(device)

    def _sum_sizes(entries: Dict[str, Optional[Dict[str, object]]]) -> Optional[int]:
        total = 0
        has = False
        for entry in entries.values():
            if entry and "size_bytes" in entry:
                total += int(entry["size_bytes"])
                has = True
        return total if has else None

    global_files = {
        "vector": _file_entry(npy_path),
        "csv": _file_entry(csv_path),
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    patch_files = {
        "vector": None,
        "csv": None,
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
        used_variant_params = dict(variant_params)

    query_config = {
        "embedding_cfg": resolved_embedding_cfg,
        "variant": variant,
        "variant_params": used_variant_params,
        "weight_id": hub_entry,
        "dataset_type": dataset_type,
        "altitude": info.altitude,
        "index": info.index,
        "prefix": info.identifier,
        "target_res": target_res,
        "query": {
            "source_file": info.source.as_posix(),
            "tag": info.tag,
            "query_dir": info.query_dir.as_posix(),
        },
    }

    global_meta = {
        "run_id": global_base,
        "token_type": "GlobalToken",
        "config": query_config,
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
            "gpu_peak_mem_mb": gpu_peak,
            "embedding_storage_bytes": _sum_sizes(global_files),
            "index_size_bytes": None,
        },
    }
    _write_meta(global_meta_path, global_meta)

    if patch_numpy is not None:
        patch_meta = {
            "run_id": patch_base,
            "token_type": "PatchToken",
            "config": query_config,
            "files": patch_files,
            "metrics": {
                "token_count": int(patch_tokens.shape[0]),
                "embedding_dim": int(patch_tokens.shape[1]) if patch_tokens.ndim == 2 else None,
                "matching_count": patch_post_info.get("kept_tokens") if patch_post_info else int(patch_tokens.shape[0]),
                "mutual_knn_tokens": patch_post_info.get("kept_tokens") if patch_post_info and variant == "mutual" else None,
                "keep_ratio": patch_post_info.get("keep_ratio") if patch_post_info else 1.0,
                "recall@1": None,
                "recall@5": None,
                "recall@10": None,
                "mAP": None,
                "top1_precision": None,
            },
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak,
                "embedding_storage_bytes": _sum_sizes(patch_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(patch_meta_path, patch_meta)


def iter_query_files() -> Iterable[Path]:
    for qdir in QUERY_DIRS:
        if not qdir.exists():
            print(f"\033[91m[WARN] Query directory missing, skipping: {qdir}\033[0m")
            continue
        for suffix in SUPPORTED_SUFFIXES:
            for path in sorted(qdir.glob(f"*{suffix}")):
                yield path


def main() -> None:
    for weight_key in VAR_WEIGHT_KEYS:
        hub_entry, weight_path, dataset_type = weights_path(weight_key)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    for weight_key in VAR_WEIGHT_KEYS:
        hub_entry, weight_path, dataset_type = weights_path(weight_key)
        print(f"[INFO] Loading model {hub_entry} ({weight_key}) on {device}")
        model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
        model.eval()

        for query_path in iter_query_files():
            info = _parse_query_filename(query_path)
            process_query_image(
                model=model,
                device=device,
                hub_entry=hub_entry,
                dataset_type=dataset_type,
                weight_key=weight_key,
                info=info,
                target_res=VAR_TARGET_RES,
                embedding_cfg=None,
                variant=VARIANT,
                variant_params=VARIANT_PARAMS,
            )
            total += 1

    print(f"\033[32m[DONE] Processed {total} query images.\033[0m")


if __name__ == "__main__":
    main()
