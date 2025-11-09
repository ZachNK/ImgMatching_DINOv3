"""
Manifest-driven runner to execute Test_Embedding / Generate_DenseFT batches.

Usage:
    python project/run_manifest.py --manifest project/json/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from Test_Embedding import run_global_embedding

try:
    from Generate_DenseFT import generate_dense_feature
except ImportError:  # pragma: no cover
    generate_dense_feature = None


BASE_DIR = Path(__file__).resolve().parent
DATA_KEY_PATH = BASE_DIR / "json/data_key.json"


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
    return parser.parse_args()


def _expand_weight_keys(raw_entry: Any) -> List[str]:
    if raw_entry is None:
        raise ValueError("\033[91m[Error] Each model entry must define 'weight_key'.\033[0m")

    raw_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
    keys: List[str] = []
    for item in raw_list:
        if not isinstance(item, str):
            raise TypeError(f"\033[91m[Error] weight_key entries must be strings, got {type(item)}\033[0m")
        for part in item.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    if not keys:
        raise ValueError("\033[91m[Error] No valid weight_key entries resolved.\033[0m")
    return keys


def _build_altitude_map(captures: Dict[str, Any]) -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = defaultdict(list)
    for capture_id, altitude in captures.items():
        mapping[int(altitude)].append(str(capture_id))
    return mapping


def _resolve_dataset_context(manifest: Dict[str, Any]) -> Tuple[str, Dict[int, List[str]], Path, str]:
    dataset_key = manifest.get("dataset_key") or os.getenv("DATASET_KEY") or _first_key(DATASETS)
    if not dataset_key or dataset_key not in DATASETS:
        raise ValueError(f"\033[91m[Error] Dataset key '{dataset_key or 'undefined'}' is not registered in data_key.json.\033[0m")

    dataset_cfg = DATASETS[dataset_key]
    captures = dataset_cfg.get("captures")
    if not isinstance(captures, dict) or not captures:
        raise ValueError(f"\033[91m[Error] Dataset '{dataset_key}' must define a 'captures' mapping.\033[0m")

    altitude_map = _build_altitude_map(captures)
    query_cfg = dataset_cfg.get("query", {})
    query_root = Path(query_cfg.get("root", "/exports"))
    query_prefix = query_cfg.get("prefix", "Q")
    return dataset_key, altitude_map, query_root, query_prefix


def _normalize_altitudes(field: Any) -> List[int]:
    if field is None:
        raise ValueError("\033[91m[Error] Image group must define 'altitudes'.\033[0m")
    altitudes = field if isinstance(field, list) else [field]
    return [int(alt) for alt in altitudes]


def _normalize_indices(field: Any, expected: int) -> List[List[int]]:
    if not field:
        raise ValueError("\033[91m[Error] Image group must define 'indices'.\033[0m")
    if isinstance(field, list) and field and isinstance(field[0], list):
        if len(field) != expected:
            raise ValueError("\033[91m[Error] indices length must match altitudes when using list-of-lists.\033[0m")
        return [[int(i) for i in lst] for lst in field]
    shared = [int(i) for i in field]
    return [shared for _ in range(expected)]


def _resolve_capture_id(altitude: int, altitude_map: Dict[int, List[str]], dataset_key: str) -> str:
    captures = altitude_map.get(int(altitude), [])
    if not captures:
        raise ValueError(f"\033[91m[Error] Altitude {altitude} is not registered under dataset '{dataset_key}'.\033[0m")
    if len(captures) > 1:
        joined = ", ".join(sorted(captures))
        raise ValueError(f"\033[91m[Error] Altitude {altitude} is ambiguous ({joined}).\033[0m")
    return captures[0]


def expand_group_entries(
    group: Dict[str, Any],
    altitude_map: Dict[int, List[str]],
    dataset_key: str,
    query_root: Path,
    query_prefix: str,
) -> List[Dict[str, Any]]:
    altitudes = _normalize_altitudes(group.get("altitudes"))
    indices_per_alt = _normalize_indices(group.get("indices"), len(altitudes))

    expanded: List[Dict[str, Any]] = []
    for altitude, idx_list in zip(altitudes, indices_per_alt):
        capture_id = _resolve_capture_id(altitude, altitude_map, dataset_key)
        name = f"{capture_id}_{int(altitude)}"
        query_dir = query_root / f"{query_prefix}{name}"
        expanded.append(
            {
                "name": name,
                "capture_id": capture_id,
                "altitude": int(altitude),
                "query_dir": query_dir.as_posix(),
                "indices": idx_list,
            }
        )
    return expanded


TOKEN_KINDS = ("global", "patch", "grid")


def _canonical_token_type(job: Dict[str, Any]) -> str:
    return str(job.get("token_type", "GlobalToken")).lower()


def _allowed_run_keys(token_type: str) -> set[str]:
    if token_type == "patchgrid":
        return {"test_embedding", "generate_denseft"}
    if token_type in {"globaltoken", "patchtoken"}:
        return {"test_embedding"}
    return {"test_embedding", "generate_denseft"}


def _ensure_valid_run_keys(job: Dict[str, Any]) -> None:
    run_cfg = job.get("run")
    if not isinstance(run_cfg, dict):
        return
    token_type = _canonical_token_type(job)
    allowed = _allowed_run_keys(token_type)
    for key in run_cfg.keys():
        if key not in allowed:
            raise ValueError(
                f"\033[91m[Error] token_type '{job.get('token_type')}' cannot set run option '{key}'.\033[0m"
            )


def _blank_output_plan() -> Dict[str, Dict[str, bool]]:
    return {key: {"npy": False, "json": False} for key in TOKEN_KINDS}


def _merge_output_plan(
    base: Dict[str, Dict[str, bool]],
    extra: Dict[str, Dict[str, bool]],
) -> None:
    for kind in TOKEN_KINDS:
        base[kind]["npy"] = base[kind]["npy"] or extra[kind]["npy"]
        base[kind]["json"] = base[kind]["json"] or extra[kind]["json"]


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


def _resolve_job_outputs(job: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    plan = _blank_output_plan()
    outputs = job.get("outputs")
    if isinstance(outputs, dict):
        for key in TOKEN_KINDS:
            if key in outputs:
                plan[key] = _parse_output_entry(outputs[key])

    if not any(plan[k]["npy"] or plan[k]["json"] for k in TOKEN_KINDS):
        token_type = str(job.get("token_type", "GlobalToken")).lower()
        if token_type == "globaltoken":
            plan["global"] = {"npy": True, "json": True}
        elif token_type == "patchtoken":
            plan["patch"] = {"npy": True, "json": True}
        elif token_type == "patchgrid":
            plan["grid"] = {"npy": True, "json": True}
        elif token_type in {"all", "alltokens"}:
            for key in TOKEN_KINDS:
                plan[key] = {"npy": True, "json": True}
        else:
            raise ValueError(f"\033[91m[Error] Unsupported token_type '{token_type}' in manifest.\033[0m")

    if not any(plan[k]["npy"] or plan[k]["json"] for k in TOKEN_KINDS):
        raise ValueError("\033[91m[Error] Each token_job must enable at least one output.\033[0m")

    return plan


def should_run(run_cfg: Dict[str, Any], key: str) -> bool:
    return bool(run_cfg.get(key, False))


def _validate_shared_test_cfg(
    anchor: Dict[str, Any],
    candidate: Dict[str, Any],
    token_label: str,
) -> None:
    label = token_label or "GlobalToken"
    for field in ("variant", "target_res", "embedding_cfg"):
        if anchor.get(field) != candidate.get(field):
            raise ValueError(
                f"\033[91m[Error] token job '{label}' must reuse the same '{field}' as other test_embedding entries.\033[0m"
            )
    if anchor.get("variant_params") != candidate.get("variant_params"):
        raise ValueError(
            f"\033[91m[Error] token job '{label}' must reuse identical variant_params when enabling test_embedding.\033[0m"
        )


def _collect_test_embedding_plan(token_jobs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    shared_cfg: Dict[str, Any] | None = None
    merged_outputs = _blank_output_plan()

    for job in token_jobs:
        _ensure_valid_run_keys(job)
        job_run = job.get("run", {})
        if not should_run(job_run, "test_embedding"):
            continue

        candidate_cfg = {
            "variant": job.get("variant", "raw"),
            "variant_params": dict(job.get("variant_params", {})),
            "target_res": int(job.get("target_res", 1024)),
            "embedding_cfg": job.get("embedding_cfg"),
        }

        if shared_cfg is None:
            shared_cfg = candidate_cfg
        else:
            _validate_shared_test_cfg(shared_cfg, candidate_cfg, str(job.get("token_type", "GlobalToken")))

        _merge_output_plan(merged_outputs, _resolve_job_outputs(job))

    if shared_cfg is None:
        return None

    if not any(merged_outputs[k]["npy"] or merged_outputs[k]["json"] for k in TOKEN_KINDS):
        return None

    return {
        **shared_cfg,
        "outputs": merged_outputs,
    }


def _should_generate_denseft(token_jobs: List[Dict[str, Any]]) -> bool:
    for job in token_jobs:
        _ensure_valid_run_keys(job)
        token_type = _canonical_token_type(job)
        if token_type != "patchgrid":
            if should_run(job.get("run", {}), "generate_denseft"):
                raise ValueError(
                    "\033[91m[Error] Only PatchGrid token jobs may enable run.generate_denseft.\033[0m"
                )
            continue
        if should_run(job.get("run", {}), "generate_denseft"):
            return True
    return False


def _build_denseft_bootstrap_plan(token_jobs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    When generate_denseft is requested without any test_embedding jobs,
    synthesize the minimal Test_Embedding configuration needed to export
    PatchGrid npy files.
    """
    bootstrap_cfg: Dict[str, Any] | None = None
    merged_outputs = _blank_output_plan()

    for job in token_jobs:
        if _canonical_token_type(job) != "patchgrid":
            continue
        if not should_run(job.get("run", {}), "generate_denseft"):
            continue

        candidate_cfg = {
            "variant": job.get("variant", "raw"),
            "variant_params": dict(job.get("variant_params", {})),
            "target_res": int(job.get("target_res", 1024)),
            "embedding_cfg": job.get("embedding_cfg"),
        }

        if bootstrap_cfg is None:
            bootstrap_cfg = candidate_cfg
        else:
            _validate_shared_test_cfg(bootstrap_cfg, candidate_cfg, str(job.get("token_type", "PatchGrid")))

        plan = _resolve_job_outputs(job)
        plan["grid"]["npy"] = True  # Ensure grid data exists for DenseFT.
        _merge_output_plan(merged_outputs, plan)

    if bootstrap_cfg is None:
        return None

    if not merged_outputs["grid"]["npy"]:
        merged_outputs["grid"]["npy"] = True

    bootstrap_cfg = {
        **bootstrap_cfg,
        "outputs": merged_outputs,
        "bootstrap": True,
    }
    return bootstrap_cfg


def execute_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_key, altitude_map, query_root, query_prefix = _resolve_dataset_context(manifest)

    for model_entry in manifest.get("models", []):
        weight_keys = _expand_weight_keys(model_entry.get("weight_key") or model_entry.get("weight_id"))
        token_jobs = model_entry.get("token_jobs", [])
        image_groups = model_entry.get("image_groups", [])

        if not token_jobs or not image_groups:
            print("\033[93m[WARN] Skipping model entry without token_jobs/image_groups.\033[0m")
            continue

        test_plan = _collect_test_embedding_plan(token_jobs)
        denseft_active = _should_generate_denseft(token_jobs)
        bootstrap_plan = None
        if test_plan is None and denseft_active:
            bootstrap_plan = _build_denseft_bootstrap_plan(token_jobs)
            test_plan = bootstrap_plan

        for group in image_groups:
            combos = expand_group_entries(group, altitude_map, dataset_key, query_root, query_prefix)
            for weight_key in weight_keys:
                if not (test_plan or denseft_active):
                    continue

                denseft_enabled = denseft_active
                if denseft_enabled and (test_plan is None or not test_plan["outputs"]["grid"]["npy"]):
                    print(
                        "    [WARN] generate_denseft requested but PatchGrid npy output is disabled; skipping dense feature job."
                    )
                    denseft_enabled = False

                if denseft_enabled and generate_dense_feature is None:
                    print("    [WARN] Generate_DenseFT module unavailable.")
                    denseft_enabled = False

                if test_plan:
                    outputs_desc = ", ".join(
                        f"{kind}=({int(test_plan['outputs'][kind]['npy'])}/{int(test_plan['outputs'][kind]['json'])})"
                        for kind in TOKEN_KINDS
                    )
                    print(
                        f"[JOB] dataset={dataset_key} weight={weight_key} token_type=CombinedTestEmbedding "
                        f"variant={test_plan['variant']} target_res={test_plan['target_res']} "
                        f"embedding_cfg={test_plan['embedding_cfg']} outputs[{outputs_desc}]"
                    )

                if denseft_enabled:
                    print(f"[JOB] dataset={dataset_key} weight={weight_key} token_type=CombinedGenerateDenseFT")

                for combo in combos:
                    altitude = combo["altitude"]
                    name = combo["name"]
                    query_dir = combo["query_dir"]

                    for index in combo["indices"]:
                        if test_plan or denseft_enabled:
                            print(f"  -> name={name} altitude={altitude} index={index} query_dir={query_dir}")

                        if test_plan:
                            run_global_embedding(
                                altitude=altitude,
                                index=index,
                                weight=weight_key,
                                target_res=test_plan["target_res"],
                                variant=test_plan["variant"],
                                embedding_cfg=test_plan["embedding_cfg"],
                                variant_params=dict(test_plan["variant_params"]),
                                output_plan=test_plan["outputs"],
                            )

                        if denseft_enabled:
                            generate_dense_feature(
                                altitude=altitude,
                                index=index,
                                weight=weight_key,
                                target_res=test_plan["target_res"],
                                variant=test_plan["variant"],
                                embedding_cfg=test_plan["embedding_cfg"],
                            )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"\033[91m[Error] Manifest not found: {manifest_path}\033[0m")
    execute_manifest(manifest_path)


if __name__ == "__main__":
    main()
