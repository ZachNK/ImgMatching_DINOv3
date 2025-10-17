#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize.py
- runCLI.py 가 생성한 JSON을 읽어 좌/우 이미지 매칭을 PNG로 그려 저장
- 입력: --pairs-root (기본 /exports/pair_match) 에 있는 JSON들
- 출력: $PAIR_VIZ_DIR/<weight>_<Aalt>_<Aframe>/<same_base>.png
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image

from imatch.env import PAIR_VIZ_DIR

PAIR_MATCH_ROOT = Path("/exports/pair_match")  # JSON 기본 루트


# ---------- 좌표 복원 유틸 ----------
def best_rect_grid(n: int) -> Tuple[int, int]:
    """
    n개 토큰을 거의 정사각형에 가깝게 채우는 (H,W) 직사각형을 찾는다.
    """
    g = int(round(np.sqrt(n)))
    if g * g == n:
        return g, g
    # 가장 가까운 인수쌍 탐색
    best = (1, n)
    best_gap = n
    for h in range(1, g + 2):
        w = int(np.ceil(n / h))
        if h * w >= n:
            gap = abs(h - w)
            if gap < best_gap:
                best_gap = gap
                best = (h, w)
    return best


def idx_to_xy(idx: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    일차원 인덱스를 (x,y) 2D grid 좌표로 변환. (x:col, y:row)
    """
    y = idx // W
    x = idx % W
    return np.stack([x, y], axis=1)  # [N,2]


def grid_to_pixels(xy: np.ndarray, img_w: int, img_h: int, W: int, H: int) -> np.ndarray:
    """
    그리드 좌표를 이미지 픽셀좌표로 스케일링
    """
    # 셀 중심 기준 매핑(시각적으로 더 자연스러움)
    px = (xy[:, 0] + 0.5) * (img_w / W)
    py = (xy[:, 1] + 0.5) * (img_h / H)
    return np.stack([px, py], axis=1)


# ---------- RANSAC ----------
def ransac_filter(ptsA: np.ndarray, ptsB: np.ndarray, method: str,
                  reproj_thresh: float = 3.0, confidence: float = 0.999, max_iters: int = 2000) -> np.ndarray:
    """
    ptsA, ptsB: [N,2] float32
    method: off | affine | homography
    return: inlier mask (bool shape [N])
    """
    N = ptsA.shape[0]
    if method == "off" or N < 4:
        return np.ones((N,), dtype=bool)

    if method == "homography" and N >= 4:
        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=reproj_thresh,
                                     maxIters=max_iters, confidence=confidence)
        if mask is None:
            return np.zeros((N,), dtype=bool)
        return mask.ravel().astype(bool)

    if method == "affine" and N >= 3:
        M, mask = cv2.estimateAffinePartial2D(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=reproj_thresh,
                                              maxIters=max_iters, confidence=confidence)
        if mask is None:
            return np.zeros((N,), dtype=bool)
        return mask.ravel().astype(bool)

    # 부족하면 전부 inlier 처리
    return np.ones((N,), dtype=bool)


# ---------- 그리기 ----------
def hstack_images(imA: np.ndarray, imB: np.ndarray, pad: int = 8, color=(30, 30, 30)) -> Tuple[np.ndarray, int]:
    """
    두 이미지를 가로로 붙이고, 사이에 패딩 컬럼 삽입.
    return: (canvas, x_offset_of_B)
    """
    h = max(imA.shape[0], imB.shape[0])
    wA, wB = imA.shape[1], imB.shape[1]
    canvas = np.full((h, wA + pad + wB, 3), color, dtype=np.uint8)
    canvas[:imA.shape[0], :wA] = imA
    canvas[:imB.shape[0], wA + pad:wA + pad + wB] = imB
    return canvas, wA + pad


def draw_matches(canvas: np.ndarray,
                 ptsA: np.ndarray, ptsB: np.ndarray,
                 xoffB: int, max_lines: int, linewidth: int,
                 draw_points: bool, alpha: int) -> None:
    """
    캔버스 위에 선/점 그리기. alpha: 0~255
    """
    N = min(max_lines, ptsA.shape[0])
    overlay = canvas.copy()
    color_line = (255, 200, 0)
    color_ptsA = (0, 220, 255)
    color_ptsB = (80, 255, 80)

    for i in range(N):
        x1, y1 = int(round(ptsA[i, 0])), int(round(ptsA[i, 1]))
        x2, y2 = int(round(ptsB[i, 0] + xoffB)), int(round(ptsB[i, 1]))
        cv2.line(overlay, (x1, y1), (x2, y2), color_line, linewidth, cv2.LINE_AA)
        if draw_points:
            cv2.circle(overlay, (x1, y1), max(1, linewidth + 1), color_ptsA, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x2, y2), max(1, linewidth + 1), color_ptsB, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha/255.0, canvas, 1.0 - alpha/255.0, 0, canvas)


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser(description="Visualize DINOv3 matches (from JSON)")
    ap.add_argument("--pairs-root", type=str, default=str(PAIR_MATCH_ROOT))
    ap.add_argument("--glob", type=str, default="**/*.json")
    ap.add_argument("--out-root", type=str, default=str(PAIR_VIZ_DIR))
    ap.add_argument("--max-lines", type=int, default=1000)
    ap.add_argument("--linewidth", type=int, default=3)
    ap.add_argument("--draw-points", action="store_true")
    ap.add_argument("--alpha", type=int, default=180, help="0-255")
    ap.add_argument("--ransac", choices=["off","affine","homography"], default="off")
    ap.add_argument("--reproj-th", type=float, default=3.0)
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    pairs_root = Path(args.pairs_root)
    out_root = Path(args.out_root)
    paths = sorted(pairs_root.glob(args.glob))
    if not paths:
        print(f"[warn] no json matched under {pairs_root} with glob={args.glob}")
        return

    for jp in paths:
        data = json.loads(jp.read_text(encoding="utf-8"))

        # 메타/이미지 경로
        imgA = Path(data["image_a"])
        imgB = Path(data["image_b"])

        # 패치/매칭 정보
        patch = data.get("patch", None)
        if not patch or not patch.get("idx_a") or not patch.get("idx_b"):
            print(f"[skip] no patch matches: {jp}")
            continue

        idx_a = np.array(patch["idx_a"], dtype=np.int64)
        idx_b = np.array(patch["idx_b"], dtype=np.int64)
        n_a = int(patch["n_a"]); n_b = int(patch["n_b"])
        g_a = patch.get("grid_g_a", None)
        g_b = patch.get("grid_g_b", None)

        # 이미지 로드
        imA = np.array(Image.open(str(imgA)).convert("RGB"))[:, :, ::-1]  # cv2 BGR 호환
        imB = np.array(Image.open(str(imgB)).convert("RGB"))[:, :, ::-1]

        # 그리드 복원 (정사각 정보가 있으면 우선, 없으면 직사각 근사)
        if g_a is not None:
            H_a, W_a = int(g_a), int(g_a)
        else:
            H_a, W_a = best_rect_grid(n_a)
        if g_b is not None:
            H_b, W_b = int(g_b), int(g_b)
        else:
            H_b, W_b = best_rect_grid(n_b)

        # 2D 좌표 → 픽셀 좌표
        xy_a = idx_to_xy(idx_a, H_a, W_a)
        xy_b = idx_to_xy(idx_b, H_b, W_b)
        ptsA = grid_to_pixels(xy_a, imA.shape[1], imA.shape[0], W_a, H_a).astype(np.float32)
        ptsB = grid_to_pixels(xy_b, imB.shape[1], imB.shape[0], W_b, H_b).astype(np.float32)

        # RANSAC 필터
        inlier = ransac_filter(ptsA, ptsB, args.ransac, args.reproj_th, args.confidence, args.iters)
        ptsA_in = ptsA[inlier]
        ptsB_in = ptsB[inlier]

        # 그리기
        canvas, xoffB = hstack_images(imA, imB)
        draw_matches(canvas, ptsA_in, ptsB_in, xoffB, args.max_lines, args.linewidth, args.draw_points, args.alpha)

        # 출력 경로: $PAIR_VIZ_DIR/<weight>_<Aalt>_<Aframe>/<same_base>.png
        # JSON 파일명과 동일 basename 사용
        rel = jp.relative_to(pairs_root)
        out_path = out_root / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        print(f"[keep] {out_path}")


if __name__ == "__main__":
    main()
