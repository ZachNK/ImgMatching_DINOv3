"""
Manifest-driven runner (query embeddings) using JSON-only configuration.

Expected manifest schema (example):
{
  "dataset_key": "shinsung_data",
  "query_embed_root": "/exports/dinov3_query_embeds",
  "experiment": {
    "variant": "sub2_pca3",
    "topk": { "use": true, "ratio": 0.05, "k": null }
  },
  "models": [
    {
      "weights": ["vitb16"],
      "target_res": 1024,
      "embedding_cfg": null,
      "image_groups": [
        { "altitudes": [200], "indices": [1], "rotation": [45, 90, 135, 180] }
      ],
      "outputs": { "global": {"npy": true,"json": true}, "patch": {...}, "grid": {...} },
      "run": { "test_embedding": true, "generate_denseft": true }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from Test_Embedding4Query import (
    QUERY_EMBED_ROOT as DEFAULT_QUERY_EMBED_ROOT,
    REPO_DIR,
    SUPPORTED_SUFFIXES,
    process_query_image,
    _parse_query_filename,
)
from Generate_DenseFT4Query import generate_query_dense_feature
from imatch.loading import (
    weights_path,
    set_dataset_key,
    normalize_group_value,
    sanitize_group_token,
    dataset_query_embed_root,
)
from imatch.pretrained import pretrained_model
from imatch.utils import progress_bar, create_progress
from variants import build_runtime_variant

TOKEN_KINDS = ("global", "patch", "grid")

BASE_DIR = Path(__file__).resolve().parent
DATA_KEY_PATH = BASE_DIR / "json/data_key.json"
QUERY_ROOT_ENV = Path(os.getenv("QUERY_ROOT", "/opt/queries"))
QUERY_PREFIX_ENV = os.getenv("QUERY_PREFIX", "Q")
QUERY_DATASET_PREFIX_ENV = os.getenv("QUERY_DATASET_PREFIX", "Q")


@dataclass
class QueryModelSession:
    model: torch.nn.Module
    hub_entry: str
    dataset_type: str
    device: torch.device
    session_id: str


_QUERY_SESSION_CACHE: Dict[Tuple[str, str, int], QueryModelSession] = {}
QUERY_SESSION_STATS: Dict[str, Dict[str, int]] = {}


def _session_cache_key(weight_key: str, device: torch.device) -> Tuple[str, str, int]:
    index = device.index if device.index is not None else -1
    return (weight_key, device.type, index)


def _query_stats_entry(weight_key: str) -> Dict[str, int]:
    entry = QUERY_SESSION_STATS.get(weight_key)
    if entry is None:
        entry = {"session_loads": 0, "direct_loads": 0, "reuses": 0}
        QUERY_SESSION_STATS[weight_key] = entry
    return entry


def collect_query_session_stats(reset: bool = False) -> Dict[str, Dict[str, int]]:
    snapshot = {weight: dict(stats) for weight, stats in QUERY_SESSION_STATS.items()}
    if reset:
        QUERY_SESSION_STATS.clear()
    return snapshot


def _clear_query_sessions() -> None:
    for session in _QUERY_SESSION_CACHE.values():
        if session.device.type == "cuda":
            torch.cuda.empty_cache()
    _QUERY_SESSION_CACHE.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query embedding jobs from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("json/manifestQuery.json"),
        help="Path to the query manifest JSON file.",
    )
    parser.add_argument(
        "--reload-each",
        action="store_true",
        help="Reload query models for every embedding instead of reusing cached sessions.",
    )
    return parser.parse_args()


def _load_data_registry() -> Dict[str, Any]:
    if not DATA_KEY_PATH.exists():
        raise FileNotFoundError(f"\033[91m[Error] data_key.json not found: {DATA_KEY_PATH}\033[0m")
    return json.loads(DATA_KEY_PATH.read_text(encoding="utf-8"))


DATA_REGISTRY = _load_data_registry()
DATASETS = DATA_REGISTRY.get("datasets", {})


def _first_key(mapping: Dict[str, Any]) -> str:
    return next(iter(mapping)) if mapping else ""


def _build_altitude_map(images: Dict[str, Any]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    for capture_id, altitude in images.items():
        mapping[normalize_group_value(altitude)].append(str(capture_id))
    return mapping


def _normalize_altitudes(field: Any) -> List[Any]:
    if field is None:
        raise ValueError("\033[91m[Error] Image group must define 'altitudes'.\033[0m")
    return list(field) if isinstance(field, list) else [field]


def _normalize_indices(field: Any, expected: int) -> List[List[int]]:
    if not field:
        raise ValueError("\033[91m[Error] Image group must define 'indices'.\033[0m")
    if isinstance(field, list) and field and isinstance(field[0], list):
        if len(field) != expected:
            raise ValueError("\033[91m[Error] indices length must match altitudes when using list-of-lists.\033[0m")
        result = []
        for lst in field:
            if len(lst) == 2 and all(isinstance(v, (int, float)) for v in lst):
                start, end = int(lst[0]), int(lst[1])
                result.append(list(range(start, end + 1)))
            else:
                result.append([int(i) for i in lst])
        return result
    if isinstance(field, list) and len(field) == 2 and all(isinstance(v, (int, float)) for v in field):
        start, end = int(field[0]), int(field[1])
        shared = list(range(start, end + 1))
    else:
        shared = [int(i) for i in field]
    return [shared for _ in range(expected)]


def _normalize_rotations(field: Any, expected: int) -> List[List[int]]:
    if not field:
        return [[0] for _ in range(expected)]
    if isinstance(field, list) and field and isinstance(field[0], list):
        if len(field) != expected:
            raise ValueError(
                "\033[91m[Error] rotation length must match altitudes when using list-of-lists.\033[0m"
            )
        return [[int(float(val)) for val in lst] for lst in field]
    shared = [int(float(val)) for val in field]
    return [shared for _ in range(expected)]


def _resolve_capture_id(altitude: Any, altitude_map: Dict[str, List[str]], dataset_key: str) -> str:
    label = normalize_group_value(altitude)
    captures = altitude_map.get(label, [])
    if not captures:
        raise ValueError(f"\033[91m[Error] Altitude/label {altitude} is not registered under dataset '{dataset_key}'.\033[0m")
    if len(captures) > 1:
        joined = ", ".join(sorted(captures))
        raise ValueError(f"\033[91m[Error] Altitude/label {altitude} is ambiguous ({joined}).\033[0m")
    return captures[0]


def _resolve_dataset_context(manifest: Dict[str, Any]) -> tuple[str, Dict[str, List[str]], Path, str, str]:
    dataset_key = manifest.get("dataset_key") or os.getenv("DATASET_KEY") or _first_key(DATASETS)
    if not dataset_key or dataset_key not in DATASETS:
        raise ValueError(
            f"\033[91m[Error] Dataset key '{dataset_key or 'undefined'}' is not registered in data_key.json.\033[0m"
        )

    dataset_cfg = DATASETS[dataset_key]
    images = dataset_cfg.get("images") or dataset_cfg.get("captures")
    if not isinstance(images, dict) or not images:
        raise ValueError(
            f"\033[91m[Error] Dataset '{dataset_key}' must define an 'images' mapping.\033[0m"
        )

    altitude_map = _build_altitude_map(images)
    query_root = QUERY_ROOT_ENV
    query_prefix = QUERY_PREFIX_ENV
    dataset_prefix = QUERY_DATASET_PREFIX_ENV
    return dataset_key, altitude_map, query_root, query_prefix, dataset_prefix


def expand_query_entries(
    group: Dict[str, Any],
    altitude_map: Dict[str, List[str]],
    dataset_key: str,
    query_root: Path,
    query_prefix: str,
    dataset_prefix: str,
) -> List[Dict[str, Any]]:
    altitudes = _normalize_altitudes(group.get("altitudes"))
    indices_per_alt = _normalize_indices(group.get("indices"), len(altitudes))
    rotations_per_alt = _normalize_rotations(group.get("rotation"), len(altitudes))

    dataset_dir = query_root / f"{dataset_prefix}{dataset_key}"
    expanded: List[Dict[str, Any]] = []
    for altitude, idx_list, rot_list in zip(altitudes, indices_per_alt, rotations_per_alt):
        capture_id = _resolve_capture_id(altitude, altitude_map, dataset_key)
        label_token = sanitize_group_token(altitude)
        label_display = normalize_group_value(altitude)
        name = f"{capture_id}_{label_token}"
        query_dir = dataset_dir / f"{query_prefix}{name}"
        expanded.append(
            {
                "name": name,
                "capture_id": capture_id,
                "altitude": label_display,
                "label_token": label_token,
                "query_dir": query_dir,
                "indices": idx_list,
                "rotations": rot_list,
            }
        )
    return expanded


def _expand_weights(raw_entry: Any) -> List[str]:
    if raw_entry is None:
        raise ValueError("\033[91m[Error] Each model entry must define 'weights'.\033[0m")
    raw_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
    keys: List[str] = []
    for item in raw_list:
        if not isinstance(item, str):
            raise TypeError(f"\033[91m[Error] weights entries must be strings, got {type(item)}\033[0m")
        for part in item.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    if not keys:
        raise ValueError("\033[91m[Error] No valid weights entries resolved.\033[0m")
    return keys


def _blank_output_plan() -> Dict[str, Dict[str, bool]]:
    return {key: {"npy": False, "json": False} for key in TOKEN_KINDS}


def _parse_output_entry(raw: Any) -> Dict[str, bool]:
    if isinstance(raw, dict):
        npy = bool(raw.get("npy"))
        json_enabled = bool(raw.get("json"))
        enable = bool(raw.get("enable")) or bool(raw.get("enabled"))
        if not (npy or json_enabled) and enable:
            npy = True
            json_enabled = True
        return {"npy": npy, "json": json_enabled}
    flag = bool(raw)
    return {"npy": flag, "json": flag}


def _normalize_outputs(outputs: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    plan = _blank_output_plan()
    if isinstance(outputs, dict):
        for key in TOKEN_KINDS:
            if key in outputs:
                plan[key] = _parse_output_entry(outputs[key])
    if not any(plan[k]["npy"] or plan[k]["json"] for k in TOKEN_KINDS):
        for key in TOKEN_KINDS:
            plan[key] = {"npy": True, "json": True}
    return plan


def _load_weighted_model(
    weight_key: str,
    device: torch.device,
    reload_each: bool = False,
) -> tuple[torch.nn.Module, str, str]:
    stats = _query_stats_entry(weight_key)
    if reload_each:
        stats["direct_loads"] += 1
        hub_entry, weight_path, dataset_type = weights_path(weight_key)
        print(
            f"[QSESSION] [DIRECT LOAD] weight={weight_key} hub_entry={hub_entry} "
            f"device={device}"
        )
        model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
        model.eval()
        return model, hub_entry, dataset_type

    cache_key = _session_cache_key(weight_key, device)
    cached = _QUERY_SESSION_CACHE.get(cache_key)
    if cached is not None:
        stats["reuses"] += 1
        print(
            f"[QSESSION] [REUSE] weight={weight_key} session={cached.session_id} device={device}"
        )
        return cached.model, cached.hub_entry, cached.dataset_type

    hub_entry, weight_path, dataset_type = weights_path(weight_key)
    print(f"[INFO] Loading model {hub_entry} ({weight_key}) on {device}")
    model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
    model.eval()
    session = QueryModelSession(
        model=model,
        hub_entry=hub_entry,
        dataset_type=dataset_type,
        device=device,
        session_id=f"{weight_key}:{id(model):x}",
    )
    _QUERY_SESSION_CACHE[cache_key] = session
    stats["session_loads"] += 1
    return session.model, session.hub_entry, session.dataset_type


def _resolve_query_matches(
    query_dir: Path,
    capture_id: str,
    label_token: str,
    index_val: int,
    rotation: int,
) -> List[Path]:
    query_dir = Path(query_dir)
    if not query_dir.exists():
        print(f"\033[91m[WARN] Query directory missing, skipping: {query_dir}\033[0m")
        return []

    stem = f"{capture_id}_{label_token}_{int(index_val):04d}_rot{int(rotation):03d}"
    matches: List[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        pattern = f"{stem}_*{suffix}"
        matches.extend(sorted(query_dir.glob(pattern)))

    # Deduplicate while preserving order.
    seen = {}
    for path in matches:
        seen[path] = None
    return list(seen.keys())


def execute_manifest(manifest_path: Path, reload_each: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_key, altitude_map, query_root, query_prefix, dataset_prefix = _resolve_dataset_context(manifest)
    set_dataset_key(dataset_key)
    base_query_root = Path(manifest.get("query_embed_root") or DEFAULT_QUERY_EMBED_ROOT)
    query_embed_root = dataset_query_embed_root(dataset_key, base_query_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = manifest.get("experiment", {}) if isinstance(manifest, dict) else {}
    variant_cfg = experiment.get("variant", {}) if isinstance(experiment, dict) else {}

    def _as_list(val: Any) -> List[Any]:
        if val is None:
            return [None]
        if isinstance(val, list):
            return list(val)
        return [val]

    raw_on = bool(variant_cfg.get("raw", True)) if isinstance(variant_cfg, dict) else True
    sub_cfg = variant_cfg.get("sub", variant_cfg.get("subsample", {})) if isinstance(variant_cfg, dict) else {}
    sub_on = bool(sub_cfg.get("use", False))
    sub_stride_list = _as_list(sub_cfg.get("stride")) if sub_on else [None]
    pca_cfg = variant_cfg.get("pca", {}) if isinstance(variant_cfg, dict) else {}
    pca_on = bool(pca_cfg.get("use", False))
    pca_dim_list = _as_list(pca_cfg.get("dim")) if pca_on else [None]

    topk_cfg = experiment.get("topk", {}) if isinstance(experiment, dict) else {}
    topk_use = bool(topk_cfg.get("use", False))
    ratio_field = topk_cfg.get("ratio")
    ratio_list = _as_list(ratio_field) if topk_use else [None]
    k_field = topk_cfg.get("k")
    k_list = _as_list(k_field) if topk_use else [None]

    pca_basis_path = experiment.get("pca_basis") or os.getenv("PCA_BASIS_PATH")

    base_variants: List[tuple[str, Optional[int]]] = []
    if raw_on:
        base_variants.append(("raw", None))
    if sub_on:
        for stride in sub_stride_list:
            base_variants.append(("subsample", stride if stride is not None else 1))

    runtime_variants = []
    for base_name, stride in base_variants:
        for pca_dim_override in pca_dim_list:
            for ratio in ratio_list:
                for k_val in k_list:
                    rv = build_runtime_variant(
                        base_name,
                        topk_enabled=topk_use,
                        topk_k=k_val,
                        topk_ratio=ratio,
                        pca_dim=pca_dim_override,
                        subsample_stride=stride,
                    )
                    runtime_variants.append(rv)

    if not runtime_variants:
        raise ValueError(
            "\033[91m[Error] No variants enabled. Set experiment.variant.raw=true or "
            "experiment.variant.sub.use=true (with stride) in manifest.\033[0m"
        )

    failures: List[Dict[str, Any]] = []
    job_counter = 0
    processed = 0
    planned_total = 0
    try:
        with create_progress() as query_progress:
            progress_task = query_progress.add_task("[cyan]Query Embedding...[/cyan]", total=0)

            for runtime_variant in runtime_variants:
                for model_entry in manifest.get("models", []):
                    weight_keys = _expand_weights(model_entry.get("weights") or model_entry.get("weight_key"))
                    image_groups = model_entry.get("image_groups", [])
                    outputs_cfg = _normalize_outputs(model_entry.get("outputs"))
                    run_cfg = model_entry.get("run", {})
                    target_res = int(model_entry.get("target_res", 1024))
                embedding_cfg = model_entry.get("embedding_cfg")
                denseft_active = bool(run_cfg.get("generate_denseft", False))
                if denseft_active:
                    outputs_cfg["grid"]["npy"] = True
                test_embedding_enabled = bool(run_cfg.get("test_embedding", True))

                if not image_groups or not test_embedding_enabled:
                    print("\033[93m[WARN] Skipping model entry without runnable image_groups/test_embedding.\033[0m")
                    continue

                expanded_groups: List[Dict[str, Any]] = []
                for group in image_groups:
                    expanded_groups.extend(
                        expand_query_entries(
                            group,
                            altitude_map,
                            dataset_key,
                            query_root,
                            query_prefix,
                            dataset_prefix,
                        )
                    )

                    for weight_key in weight_keys:
                        model, hub_entry, dataset_type = _load_weighted_model(weight_key, device, reload_each=reload_each)
                        job_counter += 1
                        print(
                            f"[JOB {job_counter}] dataset={dataset_key} weight={weight_key} variant={runtime_variant.patch_variant} "
                            f"(label={runtime_variant.label}) target_res={target_res} embedding_cfg={embedding_cfg} "
                            f"pca_dim={runtime_variant.pca_dim} stride={runtime_variant.patch_params.get('stride')} "
                            f"outputs(global/patch/grid)={[(outputs_cfg[k]['npy'], outputs_cfg[k]['json']) for k in TOKEN_KINDS]} "
                            f"denseft={int(denseft_active)}"
                        )

                        for combo in expanded_groups:
                            print(
                                f"  [Group] {combo['name']} alt={combo['altitude']} "
                                f"indices={len(combo['indices'])} rotations={combo['rotations']}"
                            )
                            for index_val in combo["indices"]:
                                for rotation in combo["rotations"]:
                                    matches = _resolve_query_matches(
                                        combo["query_dir"],
                                        combo["capture_id"],
                                        combo["label_token"],
                                        index_val,
                                        rotation,
                                    )
                                    if not matches:
                                        print(
                                            f"    [WARN] Missing files for index={index_val} rotation={rotation} "
                                            f"in {combo['query_dir']}"
                                        )
                                        continue

                                    planned_total += len(matches)
                                    query_progress.update(progress_task, total=planned_total)

                                    for query_path in matches:
                                        try:
                                            try:
                                                info = _parse_query_filename(query_path)
                                            except ValueError as err:
                                                print(f"    [WARN] Skipping unexpected file format: {query_path} -> {err}")
                                                continue

                                            print(
                                                f"    -> {query_path.name} "
                                                f"(alt={info.altitude}, idx={info.index}, rot={rotation})"
                                            )
                                            try:
                                                result = process_query_image(
                                                    model=model,
                                                    device=device,
                                                    hub_entry=hub_entry,
                                                    dataset_type=dataset_type,
                                                    weight_key=weight_key,
                                                    info=info,
                                                    target_res=target_res,
                                                    embedding_cfg=embedding_cfg,
                                                    variant=runtime_variant.patch_variant,
                                                    variant_params=dict(runtime_variant.patch_params),
                                                    output_plan=outputs_cfg,
                                                    query_embed_root=query_embed_root,
                                                    variant_label=runtime_variant.label,
                                                    topk_enabled=runtime_variant.topk_enabled,
                                                    topk_k=runtime_variant.topk_k,
                                                    topk_ratio=runtime_variant.topk_ratio,
                                                    pca_dim=runtime_variant.pca_dim,
                                                    pca_basis_path=pca_basis_path,
                                                    dataset_key=dataset_key,
                                                )
                                                processed += 1
                                                if denseft_active and result.grid_path is not None:
                                                    generate_query_dense_feature(result.grid_path)
                                                elif denseft_active:
                                                    print("      [WARN] DenseFT requested but PatchGrid npy missing; skipped.")
                                            except Exception:
                                                failure = traceback.format_exc()
                                                failures.append(
                                                    {
                                                        "weight": weight_key,
                                                        "file": query_path.as_posix(),
                                                        "rotation": rotation,
                                                        "index": index_val,
                                                        "traceback": failure,
                                                    }
                                                )
                                                print(f"      [WARN] Failed to process {query_path}")
                                                print(failure)
                                        finally:
                                            query_progress.advance(progress_task)
                        # Model stays cached for subsequent jobs; GPU memory released at shutdown.

        print(f"\n\033[32mProcessed {processed} query images across {job_counter} weight jobs.\033[0m")
        if failures:
            print("\n=== Failures ===")
            for entry in failures:
                print(
                    f"* weight={entry['weight']} file={entry['file']} "
                    f"index={entry['index']} rotation={entry['rotation']}"
                )
                print(entry["traceback"])

    finally:
        _clear_query_sessions()
        stats = collect_query_session_stats(reset=True)
        if stats:
            print("\n[QSESSION] Query weight usage summary:")
            for weight_key, data in stats.items():
                print(
                    f"  - weight={weight_key} session_loads={data.get('session_loads', 0)} "
                    f"reuses={data.get('reuses', 0)} direct_loads={data.get('direct_loads', 0)}"
                    "\n\n"
                )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"\033[91m[Error] Manifest not found: {manifest_path}\033[0m")
    execute_manifest(manifest_path, reload_each=args.reload_each)


if __name__ == "__main__":
    main()
