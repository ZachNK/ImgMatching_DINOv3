"""
Compute cosine-similarity distributions between base embeddings and rotated query embeddings.

Usage example:
    python project/analyze_rotation_similarity.py \
        --reference-dir /exports/dinov3_embeds/vitb16 \
        --query-dir /exports/dinov3_query_embeds/Qvitb16 \
        --output summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotation-wise cosine similarity analysis.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory containing base (non-rotated) embedding outputs with *_meta.json files.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory containing query (rotated) embedding outputs with *_meta.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save summary statistics (JSON).",
    )
    parser.add_argument(
        "--token-type",
        type=str,
        default="GlobalToken",
        help="Token type to analyze (default: GlobalToken).",
    )
    return parser.parse_args()


def load_vector(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return arr.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def iter_meta_files(root: Path, token_type: str) -> Iterable[Tuple[Dict, Path]]:
    for meta_path in sorted(root.rglob("*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            print(f"[WARN] Failed to parse {meta_path}: {err}")
            continue
        if meta.get("token_type") != token_type:
            continue
        vector_info = meta.get("files", {}).get("vector")
        if not vector_info:
            continue
        vector_path = meta_path.parent / vector_info["path"]
        if not vector_path.exists():
            print(f"[WARN] Vector file missing for {meta_path}")
            continue
        yield meta, vector_path


def build_reference_index(root: Path, token_type: str) -> Dict[Tuple[int, int], np.ndarray]:
    index: Dict[Tuple[int, int], np.ndarray] = {}
    for meta, vec_path in iter_meta_files(root, token_type):
        cfg = meta.get("config", {})
        altitude = int(cfg.get("altitude"))
        index_id = int(cfg.get("index"))
        key = (altitude, index_id)
        index[key] = load_vector(vec_path)
    if not index:
        raise RuntimeError(f"No reference embeddings found under {root}")
    return index


def analyze(reference_dir: Path, query_dir: Path, token_type: str) -> Dict[str, Dict[str, float]]:
    reference_map = build_reference_index(reference_dir, token_type)
    buckets: Dict[str, list] = {}
    missing_refs = 0

    for meta, vec_path in iter_meta_files(query_dir, token_type):
        cfg = meta.get("config", {})
        altitude = int(cfg.get("altitude"))
        index_id = int(cfg.get("index"))
        key = (altitude, index_id)

        if key not in reference_map:
            missing_refs += 1
            continue

        rotation_tag = cfg.get("query", {}).get("tag", "unknown")
        base_vec = reference_map[key]
        query_vec = load_vector(vec_path)
        sim = cosine_similarity(base_vec, query_vec)
        buckets.setdefault(rotation_tag, []).append(sim)

    if missing_refs:
        print(f"[WARN] {missing_refs} query samples skipped (no matching reference).")

    summary: Dict[str, Dict[str, float]] = {}
    for tag, sims in buckets.items():
        sims_arr = np.array(sims)
        summary[tag] = {
            "count": int(len(sims)),
            "mean": float(sims_arr.mean()),
            "std": float(sims_arr.std()),
            "min": float(sims_arr.min()),
            "max": float(sims_arr.max()),
        }

    return summary


def main() -> None:
    args = parse_args()
    if not args.reference_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {args.reference_dir}")
    if not args.query_dir.exists():
        raise FileNotFoundError(f"Query directory not found: {args.query_dir}")

    summary = analyze(args.reference_dir, args.query_dir, args.token_type)
    if not summary:
        print("No similarities computed (check inputs).")
        return

    print("=== Cosine similarity summary (by rotation tag) ===")
    for tag, stats in summary.items():
        print(
            f"{tag:>12}: count={stats['count']:4d}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}"
        )

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[INFO] Summary saved to {args.output}")


if __name__ == "__main__":
    main()
