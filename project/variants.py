"""
Centralised experiment variant presets and helpers.

Variants:
  - raw          : no subsampling, no PCA
  - pca3         : raw tokens but annotated with PCA dim=3 (PCA not applied here)
  - sub2         : stride-2 subsampling on the patch grid
  - sub2_pca3    : stride-2 subsampling annotated with PCA dim=3

Top-k selection is handled separately via helpers so we can toggle it on/off
without inventing additional variant names. The returned labels are meant to be
used directly in filenames/metadata to keep backbone/variant/topk combos tidy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class VariantSpec:
    name: str
    patch_variant: str  # must match imatch.postprocess variants
    patch_params: Dict[str, Any]
    pca_dim: Optional[int]
    description: str


@dataclass(frozen=True)
class RuntimeVariant:
    name: str
    label: str
    patch_variant: str
    patch_params: Dict[str, Any]
    pca_dim: Optional[int]
    topk_enabled: bool
    topk_k: Optional[int]
    topk_ratio: Optional[float]

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "patch_variant": self.patch_variant,
            "patch_params": dict(self.patch_params),
            "pca_dim": self.pca_dim,
            "topk": {
                "enabled": self.topk_enabled,
                "k": self.topk_k,
                "ratio": self.topk_ratio,
            },
        }


_VARIANT_PRESETS: Mapping[str, VariantSpec] = {
    "raw": VariantSpec(
        name="raw",
        patch_variant="raw",
        patch_params={},
        pca_dim=None,
        description="Baseline tokens without subsampling or PCA.",
    ),
    "subsample": VariantSpec(
        name="subsample",
        patch_variant="subsample",
        patch_params={"stride": 1},
        pca_dim=None,
        description="Subsampled patch grid (stride configurable via overrides).",
    ),
    "pca3": VariantSpec(
        name="pca3",
        patch_variant="raw",
        patch_params={},
        pca_dim=3,
        description="Raw tokens, annotated with PCA dim=3 (projection applied downstream).",
    ),
    "sub2": VariantSpec(
        name="sub2",
        patch_variant="subsample",
        patch_params={"stride": 2},
        pca_dim=None,
        description="Stride-2 subsampling of the patch grid.",
    ),
    "sub2_pca3": VariantSpec(
        name="sub2_pca3",
        patch_variant="subsample",
        patch_params={"stride": 2},
        pca_dim=3,
        description="Stride-2 subsampling annotated with PCA dim=3.",
    ),
}

VARIANT_NAMES = tuple(_VARIANT_PRESETS.keys())


def available_variants() -> Mapping[str, VariantSpec]:
    return _VARIANT_PRESETS


def resolve_variant(name: str) -> VariantSpec:
    key = str(name).strip().lower()
    if key not in _VARIANT_PRESETS:
        raise ValueError(f"\033[91m[Error] Unknown variant preset '{name}'. Choices: {list(_VARIANT_PRESETS)}\033[0m")
    return _VARIANT_PRESETS[key]


def _build_topk_label(topk_enabled: bool, topk_k: Optional[int], topk_ratio: Optional[float]) -> Optional[str]:
    if not topk_enabled:
        return None
    if topk_ratio is not None:
        pct = max(0.0, topk_ratio) * 100.0
        return f"top{int(round(pct)):02d}p"
    if topk_k is not None:
        return f"topk{int(topk_k)}"
    return "topk"


def build_runtime_variant(
    name: str,
    *,
    topk_enabled: bool = False,
    topk_k: Optional[int] = None,
    topk_ratio: Optional[float] = None,
    pca_dim: Optional[int] = None,
    subsample_stride: Optional[int] = None,
) -> RuntimeVariant:
    spec = resolve_variant(name)

    if topk_ratio is not None and topk_ratio <= 0:
        raise ValueError("\033[91m[Error] topk_ratio must be positive when provided.\033[0m")

    # Allow overrides for PCA dim and subsample stride to run sweeps without new names.
    patch_params = dict(spec.patch_params)
    resolved_pca_dim = spec.pca_dim if pca_dim is None else pca_dim
    if subsample_stride is not None and spec.patch_variant == "subsample":
        patch_params["stride"] = int(subsample_stride)

    topk_requested = bool(topk_enabled or topk_k is not None or topk_ratio is not None)
    topk_label = _build_topk_label(topk_requested, topk_k, topk_ratio)
    label_parts = [spec.name]
    if subsample_stride is not None and spec.patch_variant == "subsample":
        label_parts.append(f"sub{subsample_stride}")
    if resolved_pca_dim:
        label_parts.append(f"pca{resolved_pca_dim}")
    if topk_label:
        label_parts.append(topk_label)

    return RuntimeVariant(
        name=spec.name,
        label="_".join(label_parts),
        patch_variant=spec.patch_variant,
        patch_params=patch_params,
        pca_dim=resolved_pca_dim,
        topk_enabled=topk_requested,
        topk_k=int(topk_k) if topk_k is not None else None,
        topk_ratio=float(topk_ratio) if topk_ratio is not None else None,
    )
