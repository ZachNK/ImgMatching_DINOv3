"""
Build per-weight FAISS indexes from precomputed DINOv3 GlobalToken embeddings
and run TopK retrieval.
Mode keywords: 2d (query->DB), 2q (DB->query).
`--aggregate` saves a single JSON per weight when enabled.

Data layout (host paths as provided):
  Raw DB embeddings (no variant):       D:\dinov3_exports\dinov3_embeds\shinsung_data\{weight}
  Raw query embeddings (no variant):    D:\dinov3_exports\dinov3_query_embeds\shinsung_data\{weight}
  Variant DB embeddings (with variant): H:\dinov3_exports\dinov3_embeds\shinsung_data\_{weight}
  Variant query embeddings:             H:\dinov3_exports\dinov3_query_embeds\shinsung_data\_{weight}

Only GlobalToken vectors are used. Index is built per weight and searched
with the matching weight's query embeddings. Results are saved to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as err:  # pragma: no cover
    print(
        "[FATAL] faiss is not installed. Install faiss-cpu or faiss-gpu before running.\n"
        "  pip install faiss-cpu  # or faiss-gpu\n"
        f"  import error: {err}"
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Config: paths for raw vs variant embeddings
# -----------------------------------------------------------------------------
# Defaults point to in-container mount (/exports/...) but can be overridden via env.
RAW_DB_BASE = Path(os.getenv("RAW_EMBED_ROOT", "/exports/dinov3_embeds/shinsung_data"))
RAW_QUERY_BASE = Path(os.getenv("RAW_QUERY_ROOT", "/exports/dinov3_query_embeds/shinsung_data"))

RAW_DB_ROOTS: Dict[str, Path] = {
    "vits16+": RAW_DB_BASE / "vits16+",
    "vitb16": RAW_DB_BASE / "vitb16",
    "vith16+": RAW_DB_BASE / "vith16+",
    "vitl16": RAW_DB_BASE / "vitl16",
    "vitl16sat": RAW_DB_BASE / "vitl16sat",
    "vits16": RAW_DB_BASE / "vits16",
}

RAW_QUERY_ROOTS: Dict[str, Path] = {
    "vits16+": RAW_QUERY_BASE / "vits16+",
    "vitb16": RAW_QUERY_BASE / "vitb16",
    "vith16+": RAW_QUERY_BASE / "vith16+",
    "vitl16": RAW_QUERY_BASE / "vitl16",
    "vitl16sat": RAW_QUERY_BASE / "vitl16sat",
    "vits16": RAW_QUERY_BASE / "vits16",
}

# Variant roots: folder names are prefixed with an underscore.
VARIANT_DB_BASE = Path(os.getenv("VARIANT_DB_BASE", "/exports/dinov3_embeds/shinsung_data"))
VARIANT_QUERY_BASE = Path(os.getenv("VARIANT_QUERY_BASE", "/exports/dinov3_query_embeds/shinsung_data"))

# Default output root for retrieval results
DEFAULT_OUT_ROOT = Path(r"D:\dinov3_exports\dinov3_faiss_match")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@dataclass
class EmbeddingRecord:
    vec_path: Path
    meta_path: Optional[Path]
    image_path: Optional[str]


def _is_global_token(path: Path) -> bool:
    """Check filename to include only GlobalToken/QueryGlobal vectors."""
    name = path.name
    return name.startswith("GlobalToken") or name.startswith("QueryGlobal")


def _find_vectors(root: Path) -> List[EmbeddingRecord]:
    """Recursively find GlobalToken npy files under root."""
    records: List[EmbeddingRecord] = []
    for npy_path in root.rglob("*.npy"):
        if not _is_global_token(npy_path):
            continue
        meta_path = npy_path.with_name(npy_path.stem + "_meta.json")
        meta = None
        image_path = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None
        if isinstance(meta, dict):
            cfg = meta.get("config") or {}
            if isinstance(cfg, dict):
                query_info = cfg.get("query")
                if isinstance(query_info, dict) and "source_file" in query_info:
                    image_path = query_info.get("source_file")
        records.append(EmbeddingRecord(vec_path=npy_path, meta_path=meta_path if meta_path.exists() else None, image_path=image_path))
    return records


def _load_vec(path: Path) -> np.ndarray:
    arr = np.load(path)
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vec.ndim != 1:
        raise ValueError(f"Vector at {path} is not 1-D after reshape; got shape {vec.shape}")
    return vec


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows in-place for cosine/IP search."""
    faiss.normalize_L2(vecs)
    return vecs


def _build_index(db_vectors: np.ndarray, use_gpu: bool) -> faiss.Index:
    dim = db_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu:
        # Move to all available GPUs. Falls back to CPU if no GPU is present.
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception as exc:  # pragma: no cover - GPU init errors
            print(f"[WARN] GPU init failed ({exc}), falling back to CPU index.")
    index.add(db_vectors)
    return index


def _build_roots(weight: str, use_variant: bool) -> Tuple[Optional[Path], Optional[Path]]:
    if use_variant:
        return VARIANT_DB_BASE / f"_{weight}", VARIANT_QUERY_BASE / f"_{weight}"
    return RAW_DB_ROOTS.get(weight), RAW_QUERY_ROOTS.get(weight)


def _load_records(root: Optional[Path], role: str) -> Tuple[np.ndarray, List[EmbeddingRecord]]:
    if root is None or not root.exists():
        raise FileNotFoundError(f"[{role}] root not found: {root}")
    records = _find_vectors(root)
    if not records:
        raise RuntimeError(f"[{role}] No GlobalToken npy files found under {root}")
    vecs = [_load_vec(rec.vec_path) for rec in records]
    mat = np.stack(vecs, axis=0)
    _normalize(mat)
    return mat, records


def _ensure_out_dir(base: Path, weight: str, mode: str) -> Path:
    out_dir = base / f"{weight}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_single_weight(
    weight: str,
    use_variant: bool,
    k: int,
    out_root: Path,
    use_gpu: bool,
    direction: str,
    aggregate: bool,
) -> None:
    """
    direction:
      - "2d": query embeddings search DB embeddings
      - "2q": DB embeddings search query embeddings
    """
    mode = "variant" if use_variant else "raw"
    if direction not in {"2d", "2q"}:
        raise ValueError("direction must be '2d' or '2q'")

    db_root, query_root = _build_roots(weight, use_variant)
    if direction == "2d":
        index_root, search_root = db_root, query_root
        index_role, search_role = "db", "query"
    else:
        index_root, search_root = query_root, db_root
        index_role, search_role = "query", "db"

    print(f"\n[INFO] Processing weight={weight} mode={mode} direction={"query embeddings search DB embeddings" if direction == "2d" else "DB embeddings search query embeddings"}")

    idx_start = time.perf_counter()
    index_vecs, index_records = _load_records(index_root, f"{weight}:{index_role}")
    index = _build_index(index_vecs, use_gpu=use_gpu)
    index_build_ms = (time.perf_counter() - idx_start) * 1000.0

    search_vecs, search_records = _load_records(search_root, f"{weight}:{search_role}")

    out_dir = _ensure_out_dir(out_root, weight, mode)
    print(
        f"[INFO] Index vectors: {len(index_records)} ({index_role}), "
        f"Search vectors: {len(search_records)} ({search_role}), "
        f"index_build_ms={index_build_ms:.2f}"
    )

    aggregate_results: List[Dict[str, object]] = []
    total_search_ms = 0.0

    for si, srec in enumerate(search_records, start=1):
        qvec = np.asarray(search_vecs[si - 1 : si], dtype=np.float32)  # single row view
        search_start = time.perf_counter()
        k_eff = min(k, len(index_records))
        scores, idxs = index.search(qvec, k_eff)
        search_ms = (time.perf_counter() - search_start) * 1000.0
        total_search_ms += search_ms

        top_scores = scores[0].tolist()
        top_idxs = idxs[0].tolist()

        hits = []
        for rank, (db_idx, score) in enumerate(zip(top_idxs, top_scores), start=1):
            rec = index_records[db_idx]
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "vector": rec.vec_path.as_posix(),
                    "meta": rec.meta_path.as_posix() if rec.meta_path else None,
                    "image": rec.image_path,
                }
            )

        result = {
            "weight": weight,
            "mode": mode,
            "direction": direction,
            "k": k_eff,
            "index_count": len(index_records),
            "search_count": len(search_records),
            "search": {
                "vector": srec.vec_path.as_posix(),
                "meta": srec.meta_path.as_posix() if srec.meta_path else None,
                "image": srec.image_path,
            },
            "hits": hits,
            "timing_ms": {"search": search_ms},
        }

        if aggregate:
            aggregate_results.append(result)
        else:
            out_path = out_dir / f"{srec.vec_path.stem}_top{k_eff}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        if si % 25 == 0 or si == len(search_records):
            print(f"[INFO] {weight} {mode} {direction}: processed {si}/{len(search_records)}")

    if aggregate:
        agg_payload = {
            "weight": weight,
            "mode": mode,
            "direction": direction,
            "k": k,
            "index": {
                "role": index_role,
                "root": str(index_root) if index_root else None,
                "count": len(index_records),
            },
            "search": {
                "role": search_role,
                "root": str(search_root) if search_root else None,
                "count": len(search_records),
            },
            "timing_ms": {
                "index_build": index_build_ms,
                "total_search": total_search_ms,
                "avg_search": total_search_ms / len(search_records) if search_records else None,
            },
            "results": aggregate_results,
        }
        out_path = out_dir / f"{weight}_{mode}_{direction}_top{k}.json"
        out_path.write_text(json.dumps(agg_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] saved aggregate: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-weight FAISS TopK retrieval for DINOv3 embeddings.")
    parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        default=list(RAW_DB_ROOTS.keys()),
        help="Weight keys to process (default: all known raw weights).",
    )
    parser.add_argument("--variant", action="store_true", help="Use variant roots (H: with leading underscore).")
    parser.add_argument("--k", type=int, default=10, help="TopK size (default: 10).")
    parser.add_argument("--gpu", action="store_true", help="Use faiss-gpu (index_cpu_to_all_gpus).")
    parser.add_argument(
        "-m",
        "--match",
        choices=["2d", "2q"],
        default="2d",
        help="2d: 쿼리→DB 검색, 2q: DB→쿼리 검색.",
    )
    parser.add_argument(
        "-a",
        "--aggregate",
        action="store_true",
        help="Save a single aggregated JSON per weight (instead of per-search files).",
    )
    parser.add_argument(
        "-o",
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Output directory for retrieval JSON (default: {DEFAULT_OUT_ROOT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Normalize match flag aliases
    direction = args.match

    for weight in args.weights:
        try:
            run_single_weight(
                weight,
                args.variant,
                args.k,
                out_root,
                use_gpu=args.gpu,
                direction=direction,
                aggregate=bool(args.aggregate),
            )
        except Exception as exc:
            print(f"[ERROR] weight={weight} failed: {exc}")


if __name__ == "__main__":
    main()
