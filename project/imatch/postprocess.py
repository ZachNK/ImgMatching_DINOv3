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
    try:
        return VARIANT_REGISTRY[variant]
    except KeyError as err:  # pragma: no cover - defensive
        raise ValueError(f"Unknown patch-token variant '{variant}'. "
                         f"Available variants: {list(VARIANT_REGISTRY.keys())}") from err


def process_patch_tokens(
    tokens: torch.Tensor,
    variant: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply the requested post-processing variant to the patch tokens.

    Parameters
    ----------
    tokens:
        2D tensor (num_tokens, embed_dim) containing the raw patch tokens.
    variant:
        Key registered in VARIANT_REGISTRY (raw, mutual, topk, subsample, ...).
    overrides:
        Optional dictionary of parameters to override the defaults defined for
        the chosen variant.

    Returns
    -------
    processed_tokens:
        Tensor containing the filtered/processed patch tokens.
    info:
        Auxiliary metadata dict with at least ``kept_tokens``, ``keep_ratio``,
        ``params`` and optionally ``grid``/``grid_shape`` or other debug values.
    """
    processor = resolve_variant(variant)
    params = dict(processor.defaults)
    if overrides:
        params.update(overrides)

    processed_tokens, info = processor.handler(tokens, params)
    if "params" not in info:
        info["params"] = params
    return processed_tokens, info
