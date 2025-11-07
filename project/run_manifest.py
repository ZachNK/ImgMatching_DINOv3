"""
Manifest-driven runner to execute Test_Embedding / Generate_DenseFT batches.

Usage:
    python project/run_manifest.py --manifest project/json/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from Test_Embedding import run_global_embedding

try:
    from Generate_DenseFT import generate_dense_feature
except ImportError:  # pragma: no cover
    generate_dense_feature = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding jobs from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("json/manifest.json"),
        help="Path to manifest JSON file.",
    )
    return parser.parse_args()


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for data in dicts:
        if not data:
            continue
        for key, value in data.items():
            result[key] = value
    return result


def merge_variant_params(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for data in dicts:
        if not data:
            continue
        merged.update(data)
    return merged


def _expand_weight_keys(raw_entry: Any) -> List[str]:
    if raw_entry is None:
        raise ValueError("Each model entry must define 'weight_key'.")

    raw_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
    keys: List[str] = []
    for item in raw_list:
        if not isinstance(item, str):
            raise TypeError(f"weight_key entries must be strings, got {type(item)}")
        for part in item.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    if not keys:
        raise ValueError("No valid weight_key entries resolved.")
    return keys


def _normalize_name_list(group: Dict[str, Any]) -> List[str]:
    if group.get("names"):
        return [str(name) for name in group["names"]]
    if group.get("name"):
        return [str(group["name"])]
    raise ValueError("Image group must define 'names' (list) or 'name'.")


def _normalize_field_list(field: Any, fallback_len: int) -> List[Any]:
    if field is None:
        return [None] * fallback_len
    items = field if isinstance(field, list) else [field]
    if len(items) == 1 and fallback_len > 1:
        items = items * fallback_len
    if len(items) != fallback_len:
        raise ValueError("List length mismatch in image group definition.")
    return items


def expand_group_entries(group: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = _normalize_name_list(group)
    altitudes = _normalize_field_list(group.get("altitudes"), len(names))
    query_dirs = _normalize_field_list(group.get("query_dirs") or group.get("query_dir"), len(names))

    indices_raw = group.get("indices")
    if not indices_raw:
        raise ValueError("Image group must define 'indices'.")

    indices_per_name: List[List[int]]
    if isinstance(indices_raw, list) and indices_raw and isinstance(indices_raw[0], list):
        if len(indices_raw) != len(names):
            raise ValueError("indices length must match names when using list-of-lists.")
        indices_per_name = [[int(i) for i in lst] for lst in indices_raw]
    else:
        shared = [int(i) for i in indices_raw]
        indices_per_name = [shared for _ in names]

    expanded: List[Dict[str, Any]] = []
    for name, altitude, query_dir, idx_list in zip(names, altitudes, query_dirs, indices_per_name):
        expanded.append(
            {
                "name": name,
                "altitude": int(altitude),
                "query_dir": query_dir,
                "indices": idx_list,
            }
        )
    return expanded


def should_run(run_cfg: Dict[str, Any], key: str) -> bool:
    return bool(run_cfg.get(key, False))


def execute_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    defaults = manifest.get("defaults", {})
    default_run = defaults.get("run", {})
    default_variant_params = defaults.get("variant_params", {})
    default_embedding_cfg = defaults.get("embedding_cfg")
    default_variant = defaults.get("variant", "raw")
    target_res_default = defaults.get("target_res", 1024)

    for model_entry in manifest.get("models", []):
        weight_keys = _expand_weight_keys(model_entry.get("weight_key") or model_entry.get("weight_id"))
        model_run = model_entry.get("run", {})
        token_jobs = model_entry.get("token_jobs", [])
        image_groups = model_entry.get("image_groups", [])

        if not token_jobs or not image_groups:
            continue

        for weight_key in weight_keys:
            for group in image_groups:
                combos = expand_group_entries(group)
                group_run = merge_dicts(default_run, model_run, group.get("run", {}))
                group_variant_params = merge_variant_params(
                    default_variant_params,
                    model_entry.get("variant_params", {}),
                    group.get("variant_params", {}),
                )
                group_embedding_cfg = group.get("embedding_cfg") or model_entry.get("embedding_cfg") or default_embedding_cfg
                group_target_res = group.get("target_res", model_entry.get("target_res", target_res_default))

                for job in token_jobs:
                    job_run = merge_dicts(group_run, job.get("run", {}))
                    if not any(job_run.values()):
                        continue

                    variant = job.get("variant", group.get("variant", model_entry.get("variant", default_variant)))
                    variant_params = merge_variant_params(group_variant_params, job.get("variant_params", {}))
                    embedding_cfg = job.get("embedding_cfg") or group_embedding_cfg
                    token_type = job.get("token_type", "GlobalToken")

                    print(
                        f"[JOB] weight={weight_key} token_type={token_type} variant={variant} "
                        f"embedding_cfg={embedding_cfg} target_res={group_target_res}"
                    )

                    for combo in combos:
                        altitude = combo["altitude"]
                        name = combo["name"]
                        query_dir = combo.get("query_dir")

                        for index in combo["indices"]:
                            print(f"  -> name={name} altitude={altitude} index={index} query_dir={query_dir}")

                            if should_run(job_run, "test_embedding"):
                                run_global_embedding(
                                    altitude=altitude,
                                    index=index,
                                    weight=weight_key,
                                    target_res=group_target_res,
                                    variant=variant,
                                    embedding_cfg=embedding_cfg,
                                    variant_params=variant_params,
                                )

                            if should_run(job_run, "generate_denseft"):
                                if generate_dense_feature is None:
                                    print("    [WARN] Generate_DenseFT module unavailable.")
                                else:
                                    generate_dense_feature(
                                        altitude=altitude,
                                        index=index,
                                        weight=weight_key,
                                    )

                            if should_run(job_run, "run_img2denseft"):
                                print("    [INFO] run_img2denseft is legacy and not invoked by this runner.")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    execute_manifest(manifest_path)


if __name__ == "__main__":
    main()
