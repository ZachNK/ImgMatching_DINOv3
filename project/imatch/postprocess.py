"""
Patch token post-processing helpers.

The project needs to experiment with multiple filtering strategies (raw tokens,
mutual-kNN style norm thresholding, top-k selection, subsampling, etc.).
This module centralises the logic so Test_Embedding and other pipelines can
invoke the same API regardless of the selected variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch

from imatch.extracting import kp_threshold, patch2grid


ProcessorFn = Callable[[torch.Tensor, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, Any]]]


@dataclass(frozen=True)
class VariantProcessor:
    """Container describing a patch-token post-processing strategy."""

    name: str
    handler: ProcessorFn
    defaults: Dict[str, Any]
    description: str


def _ensure_tensor(tokens: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tokens):
        raise TypeError(f"tokens must be torch.Tensor, got {type(tokens)}")
    if tokens.ndim != 2:
        raise ValueError(f"patch tokens expected shape (N, C); received {tokens.shape}")
    return tokens


def _variant_raw(tokens: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokens = _ensure_tensor(tokens)
    info = {
        "kept_tokens": int(tokens.shape[0]),
        "keep_ratio": 1.0,
        "params": params,
    }
    return tokens.contiguous(), info


def _variant_mutual(tokens: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Approximate mutual k-NN filtering by discarding tokens with low L2 norm.

    This leverages the existing kp_threshold helper which normalises scores and
    keeps at least one token. The name comes from the broader project context,
    and can later be replaced by an actual mutual k-NN implementation once the
    cross-image filtering is wired up.
    """
    tokens = _ensure_tensor(tokens)
    threshold = float(params.get("norm_threshold", 0.75))
    idx_map = torch.arange(tokens.shape[0], dtype=torch.long, device=tokens.device)
    filtered_tokens, filtered_idx = kp_threshold(tokens, idx_map, threshold)

    info = {
        "kept_tokens": int(filtered_tokens.shape[0]),
        "keep_ratio": float(filtered_tokens.shape[0] / max(tokens.shape[0], 1)),
        "params": {"norm_threshold": threshold},
        "kept_indices": filtered_idx.detach().cpu().numpy(),
    }
    return filtered_tokens.contiguous(), info


def _variant_topk(tokens: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokens = _ensure_tensor(tokens)
    k = int(params.get("topk", 128))
    if k <= 0:
        raise ValueError(f"topk must be positive; received {k}")

    count = tokens.shape[0]
    if count <= k:
        info = {
            "kept_tokens": int(count),
            "keep_ratio": 1.0,
            "params": {"topk": k, "effective_topk": count},
        }
        return tokens.contiguous(), info

    norms = torch.linalg.norm(tokens, dim=1)
    topk_scores, topk_idx = torch.topk(norms, k=k, largest=True, sorted=False)
    # Sort indices to maintain deterministic ordering.
    sorted_idx, _ = torch.sort(topk_idx)
    selected = tokens.index_select(0, sorted_idx)

    info = {
        "kept_tokens": int(selected.shape[0]),
        "keep_ratio": float(selected.shape[0] / count),
        "params": {"topk": k},
        "topk_scores": topk_scores.detach().cpu().numpy(),
    }
    return selected.contiguous(), info


def _variant_subsample(tokens: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokens = _ensure_tensor(tokens)
    stride = int(params.get("stride", 2))
    if stride <= 0:
        raise ValueError(f"stride must be positive; received {stride}")

    grid = patch2grid(tokens)
    if grid.ndim != 3:
        raise ValueError(f"expected reshaped grid to be 3D, got {grid.shape}")

    subsampled = grid[::stride, ::stride, :]
    flattened = subsampled.reshape(-1, subsampled.shape[-1])
    keep_ratio = float(flattened.shape[0] / max(tokens.shape[0], 1))

    info = {
        "kept_tokens": int(flattened.shape[0]),
        "keep_ratio": keep_ratio,
        "params": {"stride": stride},
        "grid_shape": tuple(subsampled.shape),
        "grid": subsampled.contiguous(),
    }
    return flattened.contiguous(), info

def _normalize_variant_params_for_variant(
    variant: str,
    overrides: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    variant와 variant_params 조합이 규칙에 맞는지 검사하고,
    실제로 handler에 넘길 파라미터 dict를 리턴한다.

    규칙:
      - raw:
          * norm_threshold / topk / stride 모두 None 또는 아예 없는 상태여야 함
      - mutual:
          * norm_threshold 반드시 지정 (None이면 에러)
          * topk / stride 는 None 또는 키 없음이어야 함 (값 들어가면 에러)
      - topk:
          * topk 반드시 지정
          * norm_threshold / stride 금지
      - subsample:
          * stride 반드시 지정
          * norm_threshold / topk 금지
    """
    overrides = dict(overrides or {})

    allowed_keys = {"norm_threshold", "topk", "stride"}
    unknown = [k for k in overrides.keys() if k not in allowed_keys]
    if unknown:
        raise ValueError(
            f"\033[91m[Error] Unsupported keys in variant_params for variant '{variant}': "
            f"{unknown}\033[0m"
        )

    norm_threshold = overrides.get("norm_threshold")
    topk = overrides.get("topk")
    stride = overrides.get("stride")

    non_null = {
        key: value
        for key, value in (
            ("norm_threshold", norm_threshold),
            ("topk", topk),
            ("stride", stride),
        )
        if value is not None
    }

    if variant == "raw":
        # raw는 어떤 파라미터도 쓰면 안 됨
        if non_null:
            raise ValueError(
                "\033[91m[Error] variant='raw' must not specify any non-null "
                f"variant_params; got {non_null}\033[0m"
            )
        return {}

    if variant == "mutual":
        if norm_threshold is None:
            raise ValueError(
                "\033[91m[Error] variant='mutual' requires "
                "variant_params['norm_threshold'] to be set.\033[0m"
            )
        # mutual은 norm_threshold만 허용
        extra = {k: v for k, v in non_null.items() if k != "norm_threshold"}
        if extra:
            raise ValueError(
                "\033[91m[Error] variant='mutual' does not accept keys "
                f"{list(extra.keys())} in variant_params.\033[0m"
            )
        return {"norm_threshold": float(norm_threshold)}

    if variant == "topk":
        if topk is None:
            raise ValueError(
                "\033[91m[Error] variant='topk' requires "
                "variant_params['topk'] to be set.\033[0m"
            )
        extra = {k: v for k, v in non_null.items() if k != "topk"}
        if extra:
            raise ValueError(
                "\033[91m[Error] variant='topk' does not accept keys "
                f"{list(extra.keys())} in variant_params.\033[0m"
            )
        return {"topk": int(topk)}

    if variant == "subsample":
        if stride is None:
            raise ValueError(
                "\033[91m[Error] variant='subsample' requires "
                "variant_params['stride'] to be set.\033[0m"
            )
        extra = {k: v for k, v in non_null.items() if k != "stride"}
        if extra:
            raise ValueError(
                "\033[91m[Error] variant='subsample' does not accept keys "
                f"{list(extra.keys())} in variant_params.\033[0m"
            )
        return {"stride": int(stride)}

    # 여기까지 안 왔어야 함 (resolve_variant에서 이미 걸러짐)
    raise ValueError(
        f"\033[91m[Error] Unknown variant '{variant}' in "
        "_normalize_variant_params_for_variant.\033[0m"
    )


def format_variant_label(
    variant: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a human/file-friendly variant label and normalised parameters.

    Returns:
        variant_label: string appended to filenames (e.g. topk_3600, mutual_050)
        normalized_params: dict validated via _normalize_variant_params_for_variant
    """
    variant_key = str(variant).strip().lower() or "raw"
    normalized_params = _normalize_variant_params_for_variant(variant_key, overrides)

    if variant_key == "mutual":
        threshold = float(normalized_params.get("norm_threshold", 0.0))
        label = f"mutual_{int(round(threshold * 100)):03d}"
    elif variant_key == "topk":
        label = f"topk_{int(normalized_params.get('topk', 0))}"
    elif variant_key == "subsample":
        label = f"subsample_{int(normalized_params.get('stride', 0))}"
    else:
        label = variant_key

    return label, normalized_params



VARIANT_REGISTRY: Dict[str, VariantProcessor] = {
    "raw": VariantProcessor(
        name="raw",
        handler=_variant_raw,
        defaults={},
        description="No additional filtering. Save all patch tokens.",
    ),
    "mutual": VariantProcessor(
        name="mutual",
        handler=_variant_mutual,
        defaults={"norm_threshold": 0.75},
        description="Keep tokens whose L2 norm is above a normalised threshold.",
    ),
    "topk": VariantProcessor(
        name="topk",
        handler=_variant_topk,
        defaults={"topk": 128},
        description="Select top-k tokens ranked by L2 norm magnitude.",
    ),
    "subsample": VariantProcessor(
        name="subsample",
        handler=_variant_subsample,
        defaults={"stride": 2},
        description="Downsample the patch grid by a strided subsampling.",
    ),
}


def available_variants() -> Dict[str, VariantProcessor]:
    """Return the registry so other modules can list supported variants."""
    return VARIANT_REGISTRY


def resolve_variant(variant: str) -> VariantProcessor:
    """
    Resolve a variant label to a VariantProcessor.
    variant:
    - Only support'raw', 'mutual', 'topk', 'subsample'
    """
    if variant not in VARIANT_REGISTRY:
        raise ValueError(
            f"\033[91m\t[Error] Unknown patch-token variant '{variant}'. "
            f"Available variants: {list(VARIANT_REGISTRY.keys())}\033[0m"
        )
    return VARIANT_REGISTRY[variant]


def process_patch_tokens(
    tokens: torch.Tensor,
    variant: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Process patch tokens according to the specified variant and parameters.
    -----------
    inputs:
        tokens: (N, C) patch tokens
        variant: one of 'raw', 'mutual', 'topk', 'subsample'
        overrides: optional dict of parameters to override defaults for the variant

        e.g. tokens = torch.randn(1000, 256), variant='topk', overrides={'topk': 200}
    -----------
    returns:
        processed_tokens: (M, C) filtered patch tokens
        info: dict with processing details

        e.g. {'kept_tokens': 200, 'keep_ratio': 0.2, 'params': {'topk': 200}}
    -----------
    """
    processor = resolve_variant(variant)

    # manifest에서 온 variant_params를 검증/정규화
    params = _normalize_variant_params_for_variant(variant, overrides)

    processed_tokens, info = processor.handler(tokens, params)

    if "params" not in info:
        info["params"] = params
    return processed_tokens, info
