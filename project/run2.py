"""
AutoImageProcessor 기반 Hugging Face DINOv3 추출/매칭 실행 스크립트.

- 기존 run.py 와 동일한 CLI 인자를 최대한 유지하되,
  torch.hub 대신 transformers.AutoImageProcessor/AutoModel 을 사용한다.
- Hugging Face에서 제공하는 convnext 백본 모델만 alias 로 매핑되어 있으며,
  필요한 경우 --weights 에 직접 모델 ID (`facebook/...`) 를 넣어도 된다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, BatchFeature
from transformers.utils import logging as hf_logging

from imatch.cli_utils import bounded_float, bounded_int
from imatch.features import apply_keypoint_threshold, cosine_similarity
from imatch.io_images import enumerate_pairs, scan_images_by_regex
from imatch.matching import compute_matches_mutual_knn, enforce_unique_matches, grid_side, subsample_tokens
from imatch.paths import EMBED_ROOT, IMG_ROOT, MATCH_ROOT

# Hugging Face model aliases that mirror 기존 torch.hub alias 일부.
HF_MODEL_ALIASES: dict[str, str] = {
    # ConvNeXT LVD-1689M family
    "cxTiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "cxSmall": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "cxBase": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "cxLarge": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

# Light group preset to keep CLI UX 유사.
HF_MODEL_GROUPS = {
    "ConvNeXT_LVD1689M": ["cxTiny", "cxSmall", "cxBase", "cxLarge"],
}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    label: str  # filesystem friendly identifier used for export dir names


def resolve_model_specs(aliases: Sequence[str]) -> List[ModelSpec]:
    """
    aliases 목록을 Hugging Face model id 로 변환한다.
    alias 가 미리 정의되어 있지 않으면 그대로 model id 로 취급한다.
    """
    specs: List[ModelSpec] = []
    for raw in aliases:
        raw = raw.strip()
        if not raw:
            continue
        if raw in HF_MODEL_ALIASES:
            model_id = HF_MODEL_ALIASES[raw]
            label = raw
        else:
            model_id = raw
            # 파일 시스템에서 사용할 수 있도록 '/' 등을 치환
            label = raw.replace("/", "--")
        specs.append(ModelSpec(alias=raw, model_id=model_id, label=label))
    if not specs:
        raise SystemExit("No weights provided for run2.py")
    return specs


def select_aliases(args: argparse.Namespace) -> List[str]:
    """
    run.py 와 동일한 규칙으로 alias 리스트를 구성하되,
    내부 레지스트리는 HF_MODEL_ALIASES/HF_MODEL_GROUPS 를 사용한다.
    """
    if args.all_weights:
        return list(HF_MODEL_ALIASES.keys())
    if args.group:
        if args.group not in HF_MODEL_GROUPS:
            raise SystemExit(f"Unknown --group value for HF models: {args.group}")
        return HF_MODEL_GROUPS[args.group]
    if args.weights:
        return args.weights
    raise SystemExit("One of --weights / --group / --all-weights is required.")


def _load_with_token(factory, model_id: str, token: str | None, **kwargs):
    """
    transformers >= 4.37 에선 from_pretrained 가 token, <4.37 에선 use_auth_token 인자를 사용한다.
    두 경우 모두 대응하기 위해 순차적으로 시도한다.
    """
    load_kwargs = dict(kwargs)
    if not token:
        return factory.from_pretrained(model_id, **load_kwargs)

    try:
        return factory.from_pretrained(model_id, token=token, **load_kwargs)
    except TypeError:
        # 구버전 fallback
        return factory.from_pretrained(model_id, use_auth_token=token, **load_kwargs)


def prepare_inputs(
    processor: AutoImageProcessor,
    image_path: Path,
    image_size: int | None,
) -> BatchFeature:
    """
    PIL 이미지를 로드한 뒤 processor 에 맞게 전처리.
    image_size 가 지정되면 processors 의 resize 옵션에 전달한다.
    """
    img = Image.open(image_path).convert("RGB")
    kwargs = {"return_tensors": "pt"}
    if image_size:
        # 대부분의 processor 는 dict 형태(size={"shortest_edge": v}) 를 지원하지만,
        # 지원하지 않는 경우가 있으므로 예외 시 단순 정수로 재시도.
        try:
            return processor(images=img, size={"shortest_edge": int(image_size)}, **kwargs)
        except Exception:
            return processor(images=img, size=int(image_size), **kwargs)
    return processor(images=img, **kwargs)


def split_hidden_tokens(model_type: str, hidden: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state tensor 에서 CLS 토큰을 제거할 필요가 있으면 제거한다.
    convnext 계열은 CLS 토큰이 없으므로 그대로 반환한다.
    """
    if hidden.ndim != 3:
        return hidden
    hidden = hidden.squeeze(0).contiguous()
    if hidden.ndim != 2:
        return hidden
    if hidden.shape[0] <= 1:
        return hidden
    if model_type.startswith("vit") or "vit" in model_type:
        return hidden[1:, :].contiguous()
    return hidden


def ensure_float_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", dtype=torch.float32).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Matching Runner (Hugging Face AutoImageProcessor)")
    parser.add_argument("-a", "--pair-a", help="ALT.FRAME 또는 ALT (첫 번째 세트)")
    parser.add_argument("-b", "--pair-b", help="ALT.FRAME 또는 ALT (두 번째 세트)")
    parser.add_argument(
        "--regex",
        default=r".*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(jpg|jpeg|png|bmp|tif|tiff|webp)$",
        help="IMG_ROOT 아래에서 ALT.FRAME 키를 추출할 정규식",
    )
    parser.add_argument("--exts", nargs="*", default=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-w", "--weights", nargs="+", help="HF alias 또는 model id")
    group.add_argument("--group", choices=list(HF_MODEL_GROUPS.keys()))
    group.add_argument("--all-weights", action="store_true")

    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument(
        "--max-features",
        "--max-ft",
        type=bounded_int(10, 10000),
        default=1000,
        metavar="[10-10000]",
        help="매칭에 사용할 최대 패치 토큰 개수",
    )
    parser.add_argument(
        "--match-th",
        type=bounded_float(0.0, 1.0),
        default=0.1,
        metavar="[0-1]",
        help="코사인 유사도 절대 임계값",
    )
    parser.add_argument(
        "--keypoint-th",
        type=bounded_float(0.0, 1.0),
        default=0.015,
        metavar="[0-1]",
        help="패치 토큰 L2 노름 기반 필터 임계값",
    )
    parser.add_argument(
        "--line-th",
        type=bounded_float(0.0, 1.0),
        default=0.2,
        metavar="[0-1]",
        help="선형 상대 임계값 (최대 유사도의 비율)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (예: cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "-e",
        "--save-emb",
        action="store_true",
        help="각 pair 에 대해 global / patch embedding 을 EMBED_ROOT 에 저장",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face gated repo 접근 토큰 (미제공 시 환경 변수 HF_TOKEN 사용)",
    )
    parser.add_argument(
        "--quiet-hf",
        action="store_true",
        help="transformers download 로그 최소화",
    )
    args = parser.parse_args()

    if args.quiet_hf:
        hf_logging.set_verbosity_error()

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or None

    key2path = scan_images_by_regex(IMG_ROOT, args.regex, args.exts)
    keys = sorted(key2path.keys(), key=lambda s: (int(s.split(".")[0]), s.split(".")[1]))
    pairs = enumerate_pairs(keys, args.pair_a, args.pair_b)
    print(f"[images] total={len(keys)}  pairs_to_run={len(pairs)}")
    if not pairs:
        raise SystemExit("No pairs to process.")

    aliases = select_aliases(args)
    specs = resolve_model_specs(aliases)
    print(f"[models] selected={len(specs)} ids={[spec.model_id for spec in specs]}")

    if args.save_emb:
        EMBED_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA device requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = torch.device("cpu")

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda")
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode():
        with autocast_ctx:
            for spec in specs:
                print(f"[model] alias={spec.alias}  hf_id={spec.model_id}")
                try:
                    processor = _load_with_token(AutoImageProcessor, spec.model_id, hf_token)
                    model = _load_with_token(AutoModel, spec.model_id, hf_token)
                except OSError as exc:
                    msg = (
                        f"Failed to load Hugging Face model '{spec.model_id}'. "
                        "If this repository is gated, provide an access token via --hf-token or HF_TOKEN."
                    )
                    raise SystemExit(msg) from exc

                model.eval().to(device)
                model_type = getattr(model.config, "model_type", "")

                if args.save_emb:
                    (EMBED_ROOT / spec.label).mkdir(parents=True, exist_ok=True)

                for a_key, b_key in pairs:
                    pA, pB = key2path[a_key], key2path[b_key]

                    inputs_a = prepare_inputs(processor, pA, args.image_size)
                    inputs_b = prepare_inputs(processor, pB, args.image_size)
                    pixel_a = inputs_a["pixel_values"].to(device)
                    pixel_b = inputs_b["pixel_values"].to(device)

                    t0 = time.perf_counter()
                    out_a = model(pixel_values=pixel_a, return_dict=True)
                    t1 = time.perf_counter()
                    out_b = model(pixel_values=pixel_b, return_dict=True)
                    t2 = time.perf_counter()

                    fa = getattr(out_a, "pooler_output", None)
                    fb = getattr(out_b, "pooler_output", None)
                    if fa is None:
                        hidden_a = getattr(out_a, "last_hidden_state", None)
                        if hidden_a is None:
                            raise RuntimeError("Model output missing pooler_output/last_hidden_state for global feature.")
                        fa = hidden_a.mean(dim=1)
                    if fb is None:
                        hidden_b = getattr(out_b, "last_hidden_state", None)
                        if hidden_b is None:
                            raise RuntimeError("Model output missing pooler_output/last_hidden_state for global feature.")
                        fb = hidden_b.mean(dim=1)

                    fa = ensure_float_cpu(fa.squeeze(0))
                    fb = ensure_float_cpu(fb.squeeze(0))
                    cos = cosine_similarity(fa, fb)

                    patch = None
                    pa = getattr(out_a, "last_hidden_state", None)
                    pb = getattr(out_b, "last_hidden_state", None)

                    if pa is not None and pb is not None:
                        pa = split_hidden_tokens(model_type, pa)
                        pb = split_hidden_tokens(model_type, pb)
                        if pa.ndim == 2 and pb.ndim == 2:
                            orig_n_a = int(pa.shape[0])
                            orig_n_b = int(pb.shape[0])

                            ia_map = torch.arange(orig_n_a, device=pa.device, dtype=torch.long)
                            ib_map = torch.arange(orig_n_b, device=pb.device, dtype=torch.long)

                            if args.keypoint_th > 0.0:
                                pa, ia_map = apply_keypoint_threshold(pa, ia_map, args.keypoint_th)
                                pb, ib_map = apply_keypoint_threshold(pb, ib_map, args.keypoint_th)

                            if args.max_features:
                                pa, subs_a = subsample_tokens(pa, args.max_features)
                                subs_a = subs_a.to(ia_map.device, dtype=torch.long)
                                ia_map = ia_map.index_select(0, subs_a)
                                pb, subs_b = subsample_tokens(pb, args.max_features)
                                subs_b = subs_b.to(ib_map.device, dtype=torch.long)
                                ib_map = ib_map.index_select(0, subs_b)

                            pa_np = ensure_float_cpu(pa).numpy()
                            pb_np = ensure_float_cpu(pb).numpy()

                            if args.save_emb:
                                embed_dir = EMBED_ROOT / spec.label / f"{a_key}_{b_key}"
                                embed_dir.mkdir(parents=True, exist_ok=True)
                                np.save(embed_dir / "global_a.npy", fa.unsqueeze(0).numpy())
                                np.save(embed_dir / "global_b.npy", fb.unsqueeze(0).numpy())
                                np.save(embed_dir / "patch_a.npy", pa_np)
                                np.save(embed_dir / "patch_b.npy", pb_np)

                            topk_limit = int(args.max_features) if args.max_features else 400
                            ia_idx, ib_idx, sim = compute_matches_mutual_knn(pa_np, pb_np, k=1, topk=topk_limit)

                            if sim.size > 0:
                                keep = sim >= args.match_th
                                if args.line_th > 0.0:
                                    rel_min = float(sim.max()) * args.line_th
                                    keep = np.logical_and(keep, sim >= rel_min)
                                if not np.any(keep):
                                    top_idx = int(np.argmax(sim))
                                    keep = np.zeros_like(sim, dtype=bool)
                                    keep[top_idx] = True
                                ia_idx = ia_idx[keep]
                                ib_idx = ib_idx[keep]
                                sim = sim[keep]

                            ia_idx, ib_idx, sim = enforce_unique_matches(ia_idx, ib_idx, sim)

                            ia_map_cpu = ia_map.detach().to("cpu", dtype=torch.long)
                            ib_map_cpu = ib_map.detach().to("cpu", dtype=torch.long)

                            if ia_idx.size > 0:
                                ia_full = ia_map_cpu[torch.from_numpy(ia_idx)]
                                ib_full = ib_map_cpu[torch.from_numpy(ib_idx)]
                            else:
                                ia_full = torch.empty(0, dtype=torch.long)
                                ib_full = torch.empty(0, dtype=torch.long)

                            g_a = grid_side(orig_n_a)
                            g_b = grid_side(orig_n_b)

                            patch = dict(
                                n_a=orig_n_a,
                                n_b=orig_n_b,
                                n_selected_a=int(ia_map.shape[0]),
                                n_selected_b=int(ib_map.shape[0]),
                                grid_g_a=int(g_a) if g_a else None,
                                grid_g_b=int(g_b) if g_b else None,
                                idx_a=ia_full.tolist(),
                                idx_b=ib_full.tolist(),
                                similarities=sim.tolist(),
                            )

                    meta = dict(
                        img_root=str(IMG_ROOT),
                        embed_root=str(EMBED_ROOT),
                        match_root=str(MATCH_ROOT),
                        hf_model=spec.model_id,
                        device=str(device),
                        image_size=int(args.image_size),
                    )

                    payload = dict(
                        meta=meta,
                        image_a=str(pA),
                        image_b=str(pB),
                        weight=spec.label,
                        cosine=cos,
                        time_ms=dict(
                            forward_a=round((t1 - t0) * 1000, 2),
                            forward_b=round((t2 - t1) * 1000, 2),
                            total=round((t2 - t0) * 1000, 2),
                        ),
                        advanced_settings=dict(
                            match_threshold=float(args.match_th),
                            max_features=int(args.max_features),
                            keypoint_threshold=float(args.keypoint_th),
                            line_threshold=float(args.line_th),
                            matching_mode="mutual_knn_k1_unique",
                        ),
                    )
                    if patch is not None:
                        payload["patch"] = patch

                    out_dir = MATCH_ROOT / f"{spec.label}_{a_key.split(".")[0]}_{a_key.split(".")[1]}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"{spec.label}_{a_key}_{b_key}"
                    out_path = out_dir / f"{out_name}.json"
                    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
