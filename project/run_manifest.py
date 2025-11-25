"""
Manifest-driven runner (dataset embeddings) using JSON-only configuration.

Expected manifest schema (example):
{
  "dataset_key": "shinsung_data",
  "experiment": {
    "variant": "sub2_pca3",          # raw | pca3 | sub2 | sub2_pca3
    "topk": { "use": true, "ratio": 0.05, "k": null }
  },
  "models": [
    {
      "weights": ["vitb16"],
      "target_res": 1024,
      "embedding_cfg": null,
      "image_groups": [{ "altitudes": [200], "indices": [1] }],
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Test_Embedding import EmbeddingSession, collect_embedding_session_stats, run_global_embedding

try:
    from Generate_DenseFT import generate_dense_feature
except ImportError:  # pragma: no cover
    generate_dense_feature = None

from imatch.loading import set_dataset_key, normalize_group_value, sanitize_group_token
from imatch.utils import create_progress
from variants import build_runtime_variant

BASE_DIR = Path(__file__).resolve().parent
DATA_KEY_PATH = BASE_DIR / "json/data_key.json"
TOKEN_KINDS = ("global", "patch", "grid")


def _load_data_registry() -> Dict[str, Any]:
    if not DATA_KEY_PATH.exists():
        raise FileNotFoundError(f"\033[91m[Error] data_key.json not found: {DATA_KEY_PATH}\033[0m")
    return json.loads(DATA_KEY_PATH.read_text(encoding="utf-8"))


DATA_REGISTRY = _load_data_registry()
DATASETS = DATA_REGISTRY.get("datasets", {})


def _first_key(mapping: Dict[str, Any]) -> str:
    return next(iter(mapping)) if mapping else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding jobs from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("json/manifest.json"),
        help="Path to manifest JSON file.",
    )
    parser.add_argument(
        "--reload-each",
        action="store_true",
        help="Reload model weights for every embedding call instead of reusing cached models.",
    )
    return parser.parse_args()


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


def _build_altitude_map(images: Dict[str, Any]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    for capture_id, label in images.items():
        normalized = normalize_group_value(label)
        mapping[normalized].append(str(capture_id))
    return mapping


def _resolve_dataset_context(manifest: Dict[str, Any]) -> Tuple[str, Dict[str, List[str]]]:
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
    return dataset_key, altitude_map


def _normalize_altitudes(field: Any) -> List[Any]:
    if field is None:
        raise ValueError("\033[91m[Error] Image group must define 'altitudes'.\033[0m")
    return list(field) if isinstance(field, list) else [field]


def _normalize_indices(field: Any, expected: int) -> List[List[int]]:
    if not field:
        raise ValueError("\033[91m[Error] Image group must define 'indices'.\033[0m")
    # list-of-lists form
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

    # single list -> maybe range or explicit list
    if isinstance(field, list) and len(field) == 2 and all(isinstance(v, (int, float)) for v in field):
        start, end = int(field[0]), int(field[1])
        shared = list(range(start, end + 1))
    else:
        shared = [int(i) for i in field]
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


def expand_group_entries(
    group: Dict[str, Any],
    altitude_map: Dict[str, List[str]],
    dataset_key: str,
) -> List[Dict[str, Any]]:
    altitudes = _normalize_altitudes(group.get("altitudes"))
    indices_per_alt = _normalize_indices(group.get("indices"), len(altitudes))
    expanded: List[Dict[str, Any]] = []
    for altitude, idx_list in zip(altitudes, indices_per_alt):
        capture_id = _resolve_capture_id(altitude, altitude_map, dataset_key)
        label_token = sanitize_group_token(altitude)
        label_display = normalize_group_value(altitude)
        name = f"{capture_id}_{label_token}"
        expanded.append(
            {
                "name": name,
                "capture_id": capture_id,
                "altitude": label_display,
                "label_token": label_token,
                "indices": idx_list,
            }
        )
    return expanded


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


def execute_manifest(manifest_path: Path, reload_each: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_key, altitude_map = _resolve_dataset_context(manifest)
    set_dataset_key(dataset_key)

    experiment = manifest.get("experiment", {}) if isinstance(manifest, dict) else {}
    variant_cfg = experiment.get("variant", {}) if isinstance(experiment, dict) else {}

    def _as_list(val: Any) -> List[Any]:
        if val is None:
            return [None]
        if isinstance(val, list):
            return list(val)
        return [val]

    # Variant toggles
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

    planned_total = 0
    session_cache: Dict[str, EmbeddingSession] = {}
    try:
        with create_progress() as manifest_progress:
            manifest_task = manifest_progress.add_task("[cyan]Manifest [/cyan]", total=0)

            for runtime_variant in runtime_variants:
                for model_entry in manifest.get("models", []):
                    weight_keys = _expand_weights(model_entry.get("weights") or model_entry.get("weight_key"))
                    image_groups = model_entry.get("image_groups", [])
                    outputs_cfg = _normalize_outputs(model_entry.get("outputs"))
                    run_cfg = model_entry.get("run", {})
                    target_res = int(model_entry.get("target_res", 1024))
                    embedding_cfg = model_entry.get("embedding_cfg")
                    generate_denseft = bool(run_cfg.get("generate_denseft", False))
                    if generate_denseft:
                        outputs_cfg["grid"]["npy"] = True
                    test_embedding_enabled = bool(run_cfg.get("test_embedding", True))

                    if not image_groups or not test_embedding_enabled:
                        print("\033[93m[WARN] Skipping model entry without runnable image_groups/test_embedding.\033[0m")
                        continue

                    for group in image_groups:
                        combos = expand_group_entries(group, altitude_map, dataset_key)
                        group_job_size = sum(len(combo["indices"]) for combo in combos)
                        if group_job_size == 0:
                            continue

                        for weight_key in weight_keys:
                            planned_total += group_job_size
                            manifest_progress.update(manifest_task, total=planned_total)

                            print(
                                f"[JOB] dataset={dataset_key} weight={weight_key} variant={runtime_variant.patch_variant} "
                                f"(label={runtime_variant.label}) target_res={target_res} embedding_cfg={embedding_cfg} "
                                f"pca_dim={runtime_variant.pca_dim} stride={runtime_variant.patch_params.get('stride')} "
                                f"outputs(global/patch/grid)={[(outputs_cfg[k]['npy'], outputs_cfg[k]['json']) for k in TOKEN_KINDS]}"
                            )

                            session: EmbeddingSession | None = None
                            if not reload_each:
                                session = session_cache.get(weight_key)
                                if session is None:
                                    session = EmbeddingSession(weight_key)
                                    session_cache[weight_key] = session

                            for combo in combos:
                                altitude = combo["altitude"]
                                name = combo["name"]

                                for index in combo["indices"]:
                                    print(f"  -> name={name} altitude={altitude} index={index}")
                                    run_global_embedding(
                                        altitude=altitude,
                                        index=index,
                                        weight=weight_key,
                                        target_res=target_res,
                                        variant=runtime_variant.patch_variant,
                                        embedding_cfg=embedding_cfg,
                                        variant_params=dict(runtime_variant.patch_params),
                                        output_plan=outputs_cfg,
                                        dataset_key=dataset_key,
                                        session=session,
                                        variant_label=runtime_variant.label,
                                        topk_enabled=runtime_variant.topk_enabled,
                                        topk_k=runtime_variant.topk_k,
                                        topk_ratio=runtime_variant.topk_ratio,
                                        pca_dim=runtime_variant.pca_dim,
                                        pca_basis_path=pca_basis_path,
                                    )

                                    if generate_denseft and generate_dense_feature is not None:
                                        generate_dense_feature(
                                            altitude=altitude,
                                            index=index,
                                            weight=weight_key,
                                            target_res=target_res,
                                            variant=runtime_variant.patch_variant,
                                            embedding_cfg=embedding_cfg,
                                            variant_params=dict(runtime_variant.patch_params),
                                            variant_label=runtime_variant.label,
                                            pca_basis_path=pca_basis_path,
                                        )
                                    elif generate_denseft:
                                        print("    [WARN] Generate_DenseFT module unavailable.")
                                    manifest_progress.advance(manifest_task)

    finally:
        for cached_session in session_cache.values():
            cached_session.close()
        stats = collect_embedding_session_stats(reset=True)
        if stats:
            print("\n[SESSION] Embedding weight usage summary:")
            for weight_key, data in stats.items():
                print(
                    f"  - weight={weight_key} runs={data.get('runs', 0)} "
                    f"session_loads={data.get('session_loads', 0)} "
                    f"reuses={data.get('reuses', 0)} direct_loads={data.get('direct_loads', 0)}"
                )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"\033[91m[Error] Manifest not found: {manifest_path}\033[0m")
    execute_manifest(manifest_path, reload_each=args.reload_each)


if __name__ == "__main__":
    main()
