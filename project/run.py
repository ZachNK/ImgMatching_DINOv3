"""
run.py
- DINOv3 기반 이미지 매칭 배치 실행기
- 특징:
  * -a/-b 생략 시 전체 이미지 All-vs-All (N×(N-1))
  * --weights / --group / --all-weights 로 가중치 선택
  * Advanced setting: match/keypoint/line threshold + max features
  * 요청한 폴더 구조로 JSON 저장:
      /exports/dinov3_match/<weight>_<Aalt>_<Aframe>/
        <weight>_<Aalt.Aframe>_<Balt.Bframe>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from imatch.cli_utils import bounded_float, bounded_int
from imatch.features import apply_keypoint_threshold, cosine_similarity, extract_global_feature, extract_patch_tokens
from imatch.io_images import enumerate_pairs, load_image_tensor, scan_images_by_regex
from imatch.matching import compute_matches_mutual_knn, enforce_unique_matches, grid_side, subsample_tokens
from imatch.models import load_model
from imatch.paths import REPO_DIR, IMG_ROOT, EMBED_ROOT, MATCH_ROOT
from imatch.registries import WEIGHT_FILES, WEIGHT_GROUPS, resolve_weight_paths
from imatch.tfms import build_transform


def main():
    
    ## === 이미지 목록 스캔, 가중치 선택, 전처리 구축 등 초기화 ===
    p = argparse.ArgumentParser(description="DINOv3 Matching Batch Runner")
    # 이미지 선택
    p.add_argument("-a","--pair-a", help="ALT.FRAME 또는 ALT (생략시 전체)")
    p.add_argument("-b","--pair-b", help="ALT.FRAME 또는 ALT (생략시 전체)")
    p.add_argument("--regex", default=r".*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(jpg|jpeg|png|bmp|tif|tiff|webp)$", help="IMG_ROOT 하위에서 ALT.FRAME을 뽑을 정규식")
    p.add_argument("--exts", nargs="*", default=["jpg","jpeg","png","bmp","tif","tiff","webp"])
    # 가중치 선택
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-w", "--weights", nargs="+", help="가중치 옵션")
    g.add_argument("-g", "--group", choices=list(WEIGHT_GROUPS.keys()))
    g.add_argument("--all-weights", action="store_true")
    # 매칭 하이퍼파라미터
    p.add_argument("-i", "--image-size", type=int, default=336)
    p.add_argument("-x", "--max-features", type=bounded_int(10, 100000), default=2000, metavar="[10-100000]", help="패치 토큰 최대 개수")
    p.add_argument("-t", "--match-th", type=bounded_float(0.0, 1.0), default=0.05, metavar="[0-1]", help="유사도 임계값 (match threshold)")
    p.add_argument("-k", "--keypoint-th", type=bounded_float(0.0, 1.0), default=0.005, metavar="[0-1]", help="패치 토큰 보존 임계값 (keypoint threshold)")
    p.add_argument("-l", "--line-th", type=bounded_float(0.0, 1.0), default=0.2, metavar="[0-1]", help="매칭 라인 보존 임계값 (line threshold)")
    p.add_argument("-e", "--save-emb", action="store_true", help="Save global/patch embeddings to EMBED_ROOT for each pair.")
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

    if args.save_emb:
        EMBED_ROOT.mkdir(parents=True, exist_ok=True)

    ## === ※ 이미지 쌍 X 가중치 별 매칭 (DINOv3 아키텍쳐 고유 로직) ※ ===
    # 실행
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for w_alias, hub_name, ckpt in weights:
            print(f"[weight] {w_alias}  hub={hub_name}  ckpt={ckpt}")
            model, _ = load_model(REPO_DIR, args.device, hub_name, ckpt)
            if args.save_emb:
                # 가중치별 기본 임베딩 디렉터리 확보
                (EMBED_ROOT / w_alias).mkdir(parents=True, exist_ok=True)

            # 워밍업 (extract_global_feature, extract_patch_tokens 더미 호출)
            if not getattr(model, "_imatch_warmed_up", False):
                dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
                _ = extract_global_feature(model, dummy, args.device)
                _ = extract_patch_tokens(model, dummy, args.device)
                model._imatch_warmed_up = True

            # 이미지 쌍별 매칭 실행
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

    ## === DINO가 출력한 패치 임베딩을 이용한 후처리 단계 (고급 설정 적용 및 매칭 계산) ===
                    # L2 기반 키포인트 임계값 필터링
                    if args.keypoint_th > 0.0:
                        pa, ia_map = apply_keypoint_threshold(pa, ia_map, args.keypoint_th)
                        pb, ib_map = apply_keypoint_threshold(pb, ib_map, args.keypoint_th)
                    
                    # 균등 서브샘플링
                    if args.max_features:
                        pa, subs_a = subsample_tokens(pa, args.max_features)
                        subs_a = subs_a.to(ia_map.device, dtype=torch.long)
                        ia_map = ia_map.index_select(0, subs_a)

                        pb, subs_b = subsample_tokens(pb, args.max_features)
                        subs_b = subs_b.to(ib_map.device, dtype=torch.long)
                        ib_map = ib_map.index_select(0, subs_b)

                    # 상호 k-NN 매칭 (k=1)
                    pa_np = pa.detach().cpu().float().numpy()
                    pb_np = pb.detach().cpu().float().numpy()

                    if args.save_emb:
                        embed_dir = EMBED_ROOT / w_alias / f"{a_key}_{b_key}"
                        embed_dir.mkdir(parents=True, exist_ok=True)
                        # 추적용으로 글로벌/패치 임베딩을 넘파이로 저장
                        np.save(embed_dir / "global_a.npy", fa.detach().cpu().float().numpy())
                        np.save(embed_dir / "global_b.npy", fb.detach().cpu().float().numpy())
                        np.save(embed_dir / "patch_a.npy", pa_np)
                        np.save(embed_dir / "patch_b.npy", pb_np)

                    topk_limit = int(args.max_features) if args.max_features else 400
                    ia, ib, sim = compute_matches_mutual_knn(
                        pa_np,
                        pb_np,
                        k=1,
                        topk=topk_limit,
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

                    # 1:1 매칭 강제 적용
                    ia, ib, sim = enforce_unique_matches(ia, ib, sim)

                    ia_map_cpu = ia_map.detach().cpu()
                    ib_map_cpu = ib_map.detach().cpu()
                    
                    # 원본 인덱스 복원
                    ia_full = ia_map_cpu[torch.from_numpy(ia)] if ia.size > 0 else torch.empty(0, dtype=torch.long)
                    ib_full = ib_map_cpu[torch.from_numpy(ib)] if ib.size > 0 else torch.empty(0, dtype=torch.long)

                    g_a = grid_side(orig_n_a)
                    g_b = grid_side(orig_n_b)

    ## === 결과 저장: 지정된 폴더 구조로 JSON 메타데이터 생성 ===
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
                    )

                meta = dict(
                    repo_dir=str(REPO_DIR),
                    img_root=str(IMG_ROOT),
                    embed_root=str(EMBED_ROOT),
                    match_root=str(MATCH_ROOT),
                    ckpt=str(ckpt),
                    hub_model=hub_name,
                    device=args.device,
                    image_size=int(args.image_size),
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
                    matching_mode="mutual_knn_k1_unique",
                )
                if patch is not None:
                    payload["patch"] = patch

                out_dir = MATCH_ROOT / f"{w_alias}_{a_key.split(".")[0]}_{a_key.split(".")[1]}" 
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{w_alias}_{a_key}_{b_key}"
                out_path = out_dir / f"{out_name}.json"
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
