import argparse
import math
from pathlib import Path
from typing import List, Optional

import numpy as np


def _add_prefix_after_dataset(path: Path, dataset_key: str, prefix: str) -> Path:
    """
    Prefix directory components that come after the dataset key (weight/altitude/angle/token)
    to match the underscored layout used for subsampled/top-k variants on H: drive.
    """
    if not prefix:
        return path

    parts = list(path.parts)
    try:
        idx = parts.index(dataset_key)
    except ValueError:
        return path

    if idx + 1 >= len(parts):
        return path

    head = parts[: idx + 1]
    tail = parts[idx + 1 : -1]
    tail_prefixed = [p if p.startswith(prefix) else f"{prefix}{p}" for p in tail]
    return Path(*head, *tail_prefixed, parts[-1])


def _swap_root(path: Path, src_root: Path, dst_root: Path) -> Optional[Path]:
    try:
        rel = path.relative_to(src_root)
    except ValueError:
        return None
    return dst_root / rel


def resolve_path(
    path: Path,
    dataset_key: str,
    raw_root: Path,
    variant_root: Path,
    prefer_variant: bool = False,
) -> Path:
    """
    Resolve a file path across the mixed storage layout:
      - raw variants live under D:\\dinov3_exports (no leading underscores)
      - subsampled/top-k variants live under H:\\dinov3_exports with leading underscores

    The resolver tries the user-provided path first, then swaps raw<->variant roots,
    and optionally prefixes directories after the dataset key with "_" to match the
    underscored folder names.
    """
    path = path.expanduser()
    variant_hint = prefer_variant or any(
        part.startswith("_") for part in path.parts
    ) or "subsample" in path.name.lower()

    candidates: List[Path] = []

    def _add_candidate(p: Path | None) -> None:
        if p and p not in candidates:
            candidates.append(p)

    _add_candidate(path)

    for src, dst in ((raw_root, variant_root), (variant_root, raw_root)):
        swapped = _swap_root(path, src, dst)
        _add_candidate(swapped)
        if swapped and variant_hint:
            _add_candidate(_add_prefix_after_dataset(swapped, dataset_key, "_"))

    if variant_hint:
        _add_candidate(_add_prefix_after_dataset(path, dataset_key, "_"))

    for cand in candidates:
        if cand.exists():
            return cand

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"File not found. Tried:\n  {tried}")


def load_score_grid(path: Path) -> np.ndarray:
    """Load a (H, W) score grid from a .npy file."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D score grid, got shape {arr.shape} from {path}")
    return arr


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return linear indices of top-k scores in descending order."""
    flat = scores.ravel()
    k = min(k, flat.size)
    # argsort descending
    idx = np.argpartition(-flat, k - 1)[:k]
    # stable ordering (optional)
    idx = idx[np.argsort(-flat[idx])]
    return idx  # shape: (k,)


def rc_to_linear(row: np.ndarray, col: np.ndarray, H: int, W: int) -> np.ndarray:
    return row * W + col


def build_normalized_coords(H: int, W: int):
    """Return (Y, X) coordinate grids normalized to [-1, 1]."""
    # row: 0..H-1 (vertical), col: 0..W-1 (horizontal)
    ys = np.linspace(-1.0, 1.0, H)
    xs = np.linspace(-1.0, 1.0, W)
    Y, X = np.meshgrid(ys, xs, indexing="ij")  # shape (H, W)
    return Y, X


def rotate_coords(X: np.ndarray, Y: np.ndarray, angle_deg: float):
    """Rotate coordinates by angle (degrees, counter-clockwise)."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    Xr = c * X - s * Y
    Yr = s * X + c * Y
    return Xr, Yr


def map_original_to_query_indices(
    H: int,
    W: int,
    angle_deg: float,
    crop_ratio: float,
    Hq: int,
    Wq: int,
):
    """
    Build a mapping from original grid indices (r, c) to query grid indices (rq, cq),
    assuming pipeline:

      1) Original lives on square grid with normalized coords X,Y in [-1,1].
      2) Rotate image by angle_deg (counter-clockwise).
      3) In rotated coords (Xr,Yr), crop central square: |Xr| <= crop_ratio, |Yr| <= crop_ratio
         (e.g. crop_ratio=0.5 => 중앙 50% 영역).
      4) Cropped 패치를 다시 [-1,1] 전체를 채우도록 리사이즈 → query grid(Hq,Wq).

    For each original (r,c), if it lands inside the crop, map it to nearest-neighbor
    query indices (rq,cq).

    Returns:
        orig_lin   : (M,) linear indices in original grid
        query_lin  : (M,) corresponding linear indices in query grid
    """
    # 1) original normalized coords
    Y, X = build_normalized_coords(H, W)  # (H, W)

    # 2) rotate coords (counter-clockwise)
    Xr, Yr = rotate_coords(X, Y, angle_deg)

    # 3) central crop 조건
    mask_in_crop = (np.abs(Xr) <= crop_ratio) & (np.abs(Yr) <= crop_ratio)

    if not np.any(mask_in_crop):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # 4) crop 안의 좌표를 다시 [-1,1]로 normalizing
    Xq = Xr[mask_in_crop] / crop_ratio
    Yq = Yr[mask_in_crop] / crop_ratio

    # 5) [-1,1] → query index [0..Hq-1],[0..Wq-1]
    cq = np.round((Xq + 1.0) * 0.5 * (Wq - 1)).astype(np.int64)
    rq = np.round((Yq + 1.0) * 0.5 * (Hq - 1)).astype(np.int64)

    # 범위 클램프(숫자 오차 방지)
    cq = np.clip(cq, 0, Wq - 1)
    rq = np.clip(rq, 0, Hq - 1)

    # original에서 crop 안에 있는 위치 인덱스
    ro, co = np.nonzero(mask_in_crop)
    orig_lin = rc_to_linear(ro, co, H, W)
    query_lin = rc_to_linear(rq, cq, Hq, Wq)

    return orig_lin, query_lin


def topk_overlap_with_rotation(
    orig_scores: np.ndarray,
    query_scores: np.ndarray,
    angle_deg: float,
    crop_ratio: float,
    k: int,
):
    """
    Compute Top-K overlap between original and rotated+cropped query score grids.

    Steps:
      1) Original score grid에서 Top-K 위치 뽑기.
      2) 이 위치들을 회전+중앙 crop 모델로 query grid 좌표계로 매핑.
      3) Query score grid에서 Top-K 위치 뽑기.
      4) Overlap = (crop 안에 살아남은 orig-TopK 중,
                    query-TopK에도 포함된 것 개수) / K_eff

    K_eff = crop 안에 실제로 남아있는 orig-TopK 개수.
    """
    H, W = orig_scores.shape
    Hq, Wq = query_scores.shape

    # 1) Top-K in original
    orig_lin_topk = topk_indices(orig_scores, k)  # (K,)

    # 2) 전체 위치에 대한 mapping
    all_orig_lin, all_query_lin = map_original_to_query_indices(
        H, W, angle_deg, crop_ratio, Hq, Wq
    )

    if all_orig_lin.size == 0:
        return 0.0, 0, 0

    # 2-1) orig-TopK 중 crop 안에 있는 것만 선택
    mask_top = np.isin(all_orig_lin, orig_lin_topk)
    mapped_orig_lin = all_orig_lin[mask_top]
    mapped_query_lin = all_query_lin[mask_top]

    # crop 이후 실제 남은 TopK 개수
    K_eff = mapped_orig_lin.size
    if K_eff == 0:
        return 0.0, 0, len(orig_lin_topk)

    # 3) Top-K in query
    query_lin_topk = topk_indices(query_scores, k)

    # 4) overlap: mapped_query_lin vs query_lin_topk
    in_query_topk = np.isin(mapped_query_lin, query_lin_topk)
    overlap_count = int(in_query_topk.sum())
    overlap_ratio = overlap_count / float(K_eff)

    return overlap_ratio, overlap_count, K_eff


def main():
    parser = argparse.ArgumentParser(
        description="Compute rotation-aware TopK overlap between original and query score grids."
    )
    parser.add_argument(
        "--orig-score",
        type=Path,
        required=True,
        help="Path to original PatchGrid *_scores.npy",
    )
    parser.add_argument(
        "--query-score",
        type=Path,
        required=True,
        help="Path to query PatchGrid *_scores.npy",
    )
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Rotation angle in degrees (same as used for query generation).",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.5,
        help="Central crop ratio used when generating queries (e.g., 0.5 = central 50%).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3600,
        help="Top-K size to evaluate overlap.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="shinsung_data",
        help="Dataset folder name (used when resolving underscored paths).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("D:/dinov3_exports"),
        help="Base root for raw variants (default: D:/dinov3_exports).",
    )
    parser.add_argument(
        "--variant-root",
        type=Path,
        default=Path("H:/dinov3_exports"),
        help="Base root for subsampled/top-k variants (default: H:/dinov3_exports).",
    )
    parser.add_argument(
        "--prefer-variant",
        action="store_true",
        help="Prefer the variant_root + underscored layout when the initial path is missing.",
    )
    args = parser.parse_args()

    orig_path = resolve_path(
        args.orig_score,
        dataset_key=args.dataset_key,
        raw_root=args.raw_root,
        variant_root=args.variant_root,
        prefer_variant=args.prefer_variant,
    )
    query_path = resolve_path(
        args.query_score,
        dataset_key=args.dataset_key,
        raw_root=args.raw_root,
        variant_root=args.variant_root,
        prefer_variant=args.prefer_variant,
    )

    orig_scores = load_score_grid(orig_path)
    query_scores = load_score_grid(query_path)

    ratio, overlap_count, K_eff = topk_overlap_with_rotation(
        orig_scores,
        query_scores,
        angle_deg=args.angle,
        crop_ratio=args.crop_ratio,
        k=args.topk,
    )

    print("=== Top-K overlap (rotation-aware) ===")
    print(f"Original score grid : {orig_path}")
    print(f"Query    score grid : {query_path}")
    print(f"Angle (deg)         : {args.angle}")
    print(f"Crop ratio          : {args.crop_ratio}")
    print(f"Requested K         : {args.topk}")
    print(f"Effective K after crop : {K_eff}")
    print(f"Overlap count       : {overlap_count}")
    print(f"Overlap ratio       : {ratio:.4f}")


if __name__ == "__main__":
    main()
