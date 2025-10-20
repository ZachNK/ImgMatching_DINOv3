#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
runCLI.py
- DINOv3 기반 이미지 매칭 배치 실행기
- 특징:
  * -a/-b 생략 시 전체 이미지 All-vs-All (N×(N-1))
  * --weights / --group / --all-weights 로 가중치 선택
  * Advanced setting: match/keypoint/line threshold + max features
  * 요청한 폴더 구조로 JSON 저장:
      /exports/pair_match/<weight>_<Aalt>_<Aframe>/
        <weight>_<Aalt.Aframe>_<Balt.Bframe>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from imatch.env import REPO_DIR, IMG_ROOT, EXPORT_DIR, PAIR_VIZ_DIR
from imatch.io_images import load_image_tensor, scan_images_by_regex, enumerate_pairs
from imatch.tfms import build_transform
from imatch.models import load_model
from imatch.features import extract_global_feature, extract_patch_tokens, cosine_similarity
from imatch.matching import compute_matches_mutual_knn, grid_side, subsample_tokens


def bounded_float(low: float, high: float):
    """
    argparse helper to enforce a float range (inclusive).
    """
    def _validate(value: str) -> float:
        fval = float(value)
        if not (low <= fval <= high):
            raise argparse.ArgumentTypeError(f"value {fval} not in [{low}, {high}]")
        return fval
    return _validate


def bounded_int(low: int, high: int):
    """
    argparse helper to enforce an int range (inclusive).
    """
    def _validate(value: str) -> int:
        ival = int(value)
        if not (low <= ival <= high):
            raise argparse.ArgumentTypeError(f"value {ival} not in [{low}, {high}]")
        return ival
    return _validate


def apply_keypoint_threshold(tokens: torch.Tensor, idx_map: torch.Tensor, threshold: float):
    """
    Keep tokens whose normalized L2 norm exceeds the keypoint threshold.
    Falls back to keeping the strongest token if everything is filtered out.
    """
    if tokens.numel() == 0:
        return tokens, idx_map

    scores = torch.linalg.norm(tokens, dim=1)
    min_s = scores.min()
    max_s = scores.max()
    if (max_s - min_s).abs() < 1e-6:
        normalized = torch.ones_like(scores)
    else:
        normalized = (scores - min_s) / (max_s - min_s + 1e-6)

    mask = normalized >= threshold
    if not torch.any(mask):
        top_idx = torch.argmax(normalized)
        mask[top_idx] = True

    keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    tokens = tokens.index_select(0, keep_idx)
    idx_map = idx_map.index_select(0, keep_idx)
    return tokens, idx_map

# ---------- 기본값/상수 ----------
PAIR_MATCH_ROOT = Path("/exports/pair_match")  # JSON 저장 루트

# 파일명 포맷: <weight>_<Aalt>_<Aframe>/<weight>_<Aalt.Aframe>_<Balt.Bframe>.json
def split_key(key: str):
    a, b = key.split(".")
    return a, b

def out_dir_for_pair(weight_alias: str, key_a: str) -> Path:
    alt, frame = split_key(key_a)
    return PAIR_MATCH_ROOT / f"{weight_alias}_{alt}_{frame}"

def out_name_for_pair(weight_alias: str, key_a: str, key_b: str) -> str:
    return f"{weight_alias}_{key_a}_{key_b}"

# 요청하신 alias → 파일명 매핑 (세 그룹)
WEIGHT_FILES = {
    # 1) ViT_LVD-1689M
    "vit7b16":   "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "vitb16":    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vith16+":   "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "vitl16":    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "vits16":    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vits16+":   "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",

    # 2) ConvNeXT_LVD-1689M
    "cxBase":    "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "cxLarge":   "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    "cxSmall":   "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "cxTiny":    "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",

    # 3) ViT_SAT-493M
    "vit7b16sat":"dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
    "vitl16sat": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
}

# alias → torch.hub entry (모델 생성용)
HUB_BY_ALIAS = {
    "vit7b16": "dinov3_vit7b16",
    "vitb16":  "dinov3_vitb16",
    "vith16+": "dinov3_vith16plus",
    "vitl16":  "dinov3_vitl16",
    "vits16":  "dinov3_vits16",
    "vits16+": "dinov3_vits16plus",

    "cxBase":  "convnext_base",
    "cxLarge": "convnext_large",
    "cxSmall": "convnext_small",
    "cxTiny":  "convnext_tiny",

    "vit7b16sat": "dinov3_vit7b16",
    "vitl16sat":  "dinov3_vitl16",
}

# 그룹 정의
WEIGHT_GROUPS = {
    "ViT_LVD1689M":      ["vit7b16","vitb16","vith16+","vitl16","vits16","vits16+"],
    "ConvNeXT_LVD1689M": ["cxBase","cxLarge","cxSmall","cxTiny"],
    "ViT_SAT493M":       ["vit7b16sat","vitl16sat"],
}


def resolve_weight_paths(aliases, roots):
    """
    aliases 리스트를 실제 ckpt 파일 경로로 해석
    roots: /opt/weights, /opt/weights/01_..., 02_..., 03_... 등
    """
    out = []
    for alias in aliases:
        if alias not in WEIGHT_FILES:
            raise SystemExit(f"Unknown weight alias: {alias}")
        fname = WEIGHT_FILES[alias]
        found = None
        for r in roots:
            cand = Path(r) / fname
            if cand.is_file():
                found = cand
                break
        if not found:
            raise SystemExit(f"[ckpt] not found for {alias}: {fname}")
        hub_name = HUB_BY_ALIAS.get(alias, "dinov3_vitl16")
        out.append((alias, hub_name, found))
    return out


def main():
    p = argparse.ArgumentParser(description="DINOv3 Matching Batch Runner")
    # 이미지 선택
    p.add_argument("-a","--pair-a", help="ALT.FRAME 또는 ALT (생략시 전체)")
    p.add_argument("-b","--pair-b", help="ALT.FRAME 또는 ALT (생략시 전체)")
    p.add_argument("--regex", default=r".*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(jpg|jpeg|png|bmp|tif|tiff|webp)$",
                   help="IMG_ROOT 하위에서 ALT.FRAME을 뽑을 정규식")
    p.add_argument("--exts", nargs="*", default=["jpg","jpeg","png","bmp","tif","tiff","webp"])
    # 가중치 선택
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-w", "--weights", nargs="+", help="alias 리스트")
    g.add_argument("--group", choices=list(WEIGHT_GROUPS.keys()))
    g.add_argument("--all-weights", action="store_true")
    # 매칭 하이퍼파라미터
    p.add_argument("--image-size", type=int, default=336)
    p.add_argument("--mutual-k", type=int, default=10)
    p.add_argument("--topk", type=int, default=400)
    p.add_argument("--max-features", "--max-ft", type=bounded_int(10, 10000), default=1000,
                   metavar="[10-10000]", help="패치 토큰 최대 개수")
    p.add_argument("--match-th", type=bounded_float(0.0, 1.0), default=0.1,
                   metavar="[0-1]", help="유사도 임계값 (match threshold)")
    p.add_argument("--keypoint-th", type=bounded_float(0.0, 1.0), default=0.015,
                   metavar="[0-1]", help="패치 토큰 보존 임계값 (keypoint threshold)")
    p.add_argument("--line-th", type=bounded_float(0.0, 1.0), default=0.2,
                   metavar="[0-1]", help="매칭 라인 보존 임계값 (line threshold)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # 이미지 후보 스캔
    key2path = scan_images_by_regex(IMG_ROOT, args.regex, args.exts)
    keys = sorted(key2path.keys(), key=lambda s: (int(s.split(".")[0]), s.split(".")[1]))
    pairs = enumerate_pairs(keys, args.pair_a, args.pair_b)
    print(f"[images] total={len(keys)}  pairs_to_run={len(pairs)}")

    # 가중치 해석
    weight_aliases = (list(WEIGHT_FILES.keys()) if args.all_weights
                      else WEIGHT_GROUPS[args.group] if args.group
                      else args.weights)
    weight_roots = [
        Path("/opt/weights"),
        Path("/opt/weights/01_ViT_LVD-1689M"),
        Path("/opt/weights/02_ConvNeXT_LVD-1689M"),
        Path("/opt/weights/03_ViT_SAT-493M"),
    ]
    weights = resolve_weight_paths(weight_aliases, weight_roots)
    print(f"[weights] selected={len(weights)} → {[w[0] for w in weights]}")

    # 전처리
    tfm = build_transform(args.image_size)

    # 실행
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for w_alias, hub_name, ckpt in weights:
            print(f"[weight] {w_alias}  hub={hub_name}  ckpt={ckpt}")
            model, _ = load_model(REPO_DIR, args.device, hub_name, ckpt)

            for a_key, b_key in pairs:
                pA, pB = key2path[a_key], key2path[b_key]
                xa = tfm(load_image_tensor(pA)).unsqueeze(0)
                xb = tfm(load_image_tensor(pB)).unsqueeze(0)

                t0 = time.perf_counter()
                fa = extract_global_feature(model, xa, args.device); t1 = time.perf_counter()
                fb = extract_global_feature(model, xb, args.device); t2 = time.perf_counter()
                pa = extract_patch_tokens(model, xa, args.device)
                pb = extract_patch_tokens(model, xb, args.device)

                cos = cosine_similarity(fa, fb)

                patch = None
                if (pa is not None) and (pb is not None):
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

                    pa_np = pa.detach().cpu().float().numpy()
                    pb_np = pb.detach().cpu().float().numpy()

                    ia, ib, sim = compute_matches_mutual_knn(
                        pa_np,
                        pb_np,
                        k=args.mutual_k, topk=args.topk
                    )

                    if sim.size > 0:
                        keep = sim >= args.match_th
                        if sim.size > 0 and args.line_th > 0.0:
                            rel_min = sim.max() * args.line_th
                            keep = np.logical_and(keep, sim >= rel_min)
                        if not np.any(keep):
                            top_idx = int(np.argmax(sim))
                            keep = np.zeros_like(sim, dtype=bool)
                            keep[top_idx] = True
                        ia = ia[keep]
                        ib = ib[keep]
                        sim = sim[keep]

                    ia_map_cpu = ia_map.detach().cpu()
                    ib_map_cpu = ib_map.detach().cpu()

                    ia_full = ia_map_cpu[torch.from_numpy(ia)] if ia.size > 0 else torch.empty(0, dtype=torch.long)
                    ib_full = ib_map_cpu[torch.from_numpy(ib)] if ib.size > 0 else torch.empty(0, dtype=torch.long)

                    g_a = grid_side(orig_n_a)
                    g_b = grid_side(orig_n_b)

                    patch = dict(
                        n_a=orig_n_a,
                        n_b=orig_n_b,
                        n_selected_a=int(ia_map.shape[0]),
                        n_selected_b=int(ib_map.shape[0]),
                        grid_g_a=(int(g_a) if g_a else None),
                        grid_g_b=(int(g_b) if g_b else None),
                        idx_a=ia_full.tolist(),
                        idx_b=ib_full.tolist(),
                        similarities=sim.tolist(),
                        topk=int(args.topk),
                        mutual_k=int(args.mutual_k),
                        match_threshold=float(args.match_th),
                        keypoint_threshold=float(args.keypoint_th),
                        line_threshold=float(args.line_th),
                        max_features=int(args.max_features),
                    )

                meta = dict(
                    repo_dir=str(REPO_DIR),
                    img_root=str(IMG_ROOT),
                    export_root=str(EXPORT_DIR),
                    ckpt=str(ckpt),
                    hub_model=hub_name,
                    device=args.device,
                    image_size=int(args.image_size),
                    pair_match_root=str(PAIR_MATCH_ROOT),
                )
                payload = dict(
                    meta=meta, image_a=str(pA), image_b=str(pB),
                    weight=w_alias, cosine=cos,
                    time_ms=dict(
                        forward_a=round((t1 - t0) * 1000, 2),
                        forward_b=round((t2 - t1) * 1000, 2),
                        total=round((t2 - t0) * 1000, 2),
                    ),
                )
                payload["advanced_settings"] = dict(
                    match_threshold=float(args.match_th),
                    max_features=int(args.max_features),
                    keypoint_threshold=float(args.keypoint_th),
                    line_threshold=float(args.line_th),
                    mutual_k=int(args.mutual_k),
                    topk=int(args.topk),
                )
                if patch is not None:
                    payload["patch"] = patch

                out_dir = out_dir_for_pair(w_alias, a_key)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = out_name_for_pair(w_alias, a_key, b_key)
                out_path = out_dir / f"{out_name}.json"
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
