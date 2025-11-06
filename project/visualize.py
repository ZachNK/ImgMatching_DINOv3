"""
visualize.py (interactive 확장, 최종본)
- 원래: 폴더를 고르면 그 폴더 내 모든 JSON 시각화
- 추가:
  1) 폴더 선택 후, 해당 폴더 내 JSON을 다시 번호로 보여주고 '단일 JSON만' 시각화 가능
  2) 최상위 프롬프트에서 '-a' 입력 시, 루트 하위 모든 폴더의 모든 JSON 일괄 시각화
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Optional
from PIL import Image
from imatch.loading import MATCH_ROOT, VIS_ROOT

# ---------- OpenCV RANSAC 플래그 탐색 ----------
def _discover_homography_methods() -> Tuple[dict, str]:
    candidates = {
        "cv2_ransac": ["RANSAC"],
        "cv2_usac_default": ["USAC_DEFAULT"],
        "cv2_usac_parallel": ["USAC_PARALLEL"],
        "cv2_usac_accurate": ["USAC_ACCURATE"],
        "cv2_usac_fast": ["USAC_FAST"],
        "cv2_usac_prosac": ["USAC_PROSAC"],
        "cv2_usac_magsac": ["USAC_MAGSAC"],
        "cv2_usac_fm_8pts": ["USAC_FM_8PTS"],
        "poselib": ["USAC_POSE", "USAC_POSELIB", "USAC_RELATIVE_POSE"],
    }
    mapping = {}
    for name, attr_names in candidates.items():
        for attr in attr_names:
            flag = getattr(cv2, attr, None)
            if isinstance(flag, int):
                mapping[name] = flag
                break
    default_name = "cv2_usac_magsac" if "cv2_usac_magsac" in mapping else "cv2_ransac"
    if default_name not in mapping and mapping:
        default_name = next(iter(mapping.keys()))
    return mapping, default_name


HOMOGRAPHY_METHODS, HOMOGRAPHY_DEFAULT = _discover_homography_methods()


# ---------- 선택 리스트 ----------
def list_pick_candidates(root: Path) -> List[str]:
    candidates: List[str] = []
    if not root.exists():
        return candidates
    for entry in sorted(root.iterdir()):
        rel = entry.name
        if entry.is_dir():
            candidates.append(f"{rel}/")
        elif entry.is_file() and entry.suffix.lower() == ".json":
            candidates.append(rel)
    return candidates


def _list_json_in_folder(folder: Path) -> List[str]:
    """폴더 내 JSON을 재귀로 수집하여 폴더 기준 상대경로 문자열 리스트 반환."""
    if not folder.exists():
        return []
    rels: List[str] = []
    for jp in sorted(folder.rglob("*.json")):
        try:
            rels.append(str(jp.relative_to(folder)))
        except Exception:
            rels.append(jp.name)
    return rels


def prompt_pick_top(root: Path) -> List[str]:
    """
    1차 프롬프트:
      - 숫자 입력: 해당 항목 선택
      - '-a' 입력: 루트 하위 전체 JSON(**/*.json) 일괄 처리
      - 엔터: 재입력 유도
    반환: ["**/*.json"] 또는 ["선택폴더/"] 또는 ["선택.json"]
    """
    candidates = list_pick_candidates(root)

    if not sys.stdin.isatty():
        # non-tty에서는 전체 처리
        return ["**/*.json"]

    if not candidates:
        print("[info] 후보가 없어 전체 JSON을 시각화합니다.")
        return ["**/*.json"]

    print("번호로 선택하시오 (또는 -a = 전체일괄):")
    for idx, rel in enumerate(candidates, start=1):
        print(f"  {idx:2d}. {rel}")
    while True:
        choice = input("폴더/파일 번호 또는 '-a': ").strip()
        if choice == "-a":
            return ["**/*.json"]
        if not choice:
            print("  숫자 또는 '-a'를 입력하세요.")
            continue
        if not choice.isdigit():
            print("  잘못된 입력입니다. 숫자 또는 '-a'만 허용됩니다.")
            continue
        idx = int(choice) - 1
        if not 0 <= idx < len(candidates):
            print(f"  1에서 {len(candidates)} 사이의 숫자를 입력하세요.")
            continue
        return [candidates[idx]]


def prompt_pick_json_in_folder(root: Path, picked: str) -> List[str]:
    """
    2차 프롬프트(폴더 선택 시):
      - 숫자: 해당 JSON '한 개'만 시각화
      - 엔터: 폴더 내 전체 JSON 일괄 처리(원래 방식)
    반환: ["폴더/내/선택.json"] 또는 ["폴더/"]
    """
    folder = (root / picked.rstrip("/"))
    json_list = _list_json_in_folder(folder)

    if not json_list:
        print("[info] 해당 폴더에 JSON이 없어 스킵합니다.")
        return []

    print(f"'{picked}' 폴더 내 JSON 목록:")
    for idx, rel in enumerate(json_list, start=1):
        print(f"  {idx:2d}. {rel}")
    choice = input("단일 JSON만 시각화하려면 번호 입력, 엔터=폴더 전체: ").strip()
    if not choice:
        return [picked]
    if choice.isdigit():
        jdx = int(choice) - 1
        if 0 <= jdx < len(json_list):
            return [str(Path(picked) / json_list[jdx])]
        else:
            print("[warn] 번호 범위 초과. 폴더 전체를 시각화합니다.")
            return [picked]
    else:
        print("[warn] 잘못된 입력. 폴더 전체를 시각화합니다.")
        return [picked]


def add_json_path(acc: List[Path], seen: Set[Path], path: Path) -> None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    acc.append(resolved)


def collect_jsons(root: Path, selections: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()
    for sel in selections:
        if not sel:
            continue
        target = (root if sel in (".", "/") else root / sel)
        # 글롭 패턴 처리
        if any(ch in sel for ch in ["*", "?", "["]) and not target.exists():
            for jp in sorted(root.rglob(sel)):
                add_json_path(files, seen, jp)
            continue
        if not target.exists():
            print(f"[warn] 선택한 항목이 존재하지 않습니다: {sel}")
            continue
        if target.is_dir():
            for jp in sorted(target.rglob("*.json")):
                add_json_path(files, seen, jp)
        else:
            add_json_path(files, seen, target)
    return sorted(files)


# ---------- 좌표 복원 유틸 ----------
def best_rect_grid(n: int) -> Tuple[int, int]:
    g = int(round(np.sqrt(n)))
    if g * g == n:
        return g, g
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
    # row-major: y(행), x(열)
    y = idx // W
    x = idx % W
    return np.stack([x, y], axis=1)


def grid_to_pixels(xy: np.ndarray, img_w: int, img_h: int, W: int, H: int) -> np.ndarray:
    # 셀 중심 좌표로 매핑
    px = (xy[:, 0] + 0.5) * (img_w / W)
    py = (xy[:, 1] + 0.5) * (img_h / H)
    return np.stack([px, py], axis=1)


# ---------- RANSAC ----------
def ransac_filter(
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    method: str,
    homography_flag: int,
    reproj_thresh: float,
    confidence: float,
    max_iters: int,
) -> np.ndarray:
    N = min(ptsA.shape[0], ptsB.shape[0])
    if N == 0 or method == "off":
        return np.ones((N,), dtype=bool)

    if method == "homography" and N >= 4:
        _, mask = cv2.findHomography(
            ptsA, ptsB, homography_flag,
            ransacReprojThreshold=reproj_thresh,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is None:
            return np.zeros((N,), dtype=bool)
        return mask.ravel().astype(bool)[:N]

    if method == "affine" and N >= 3:
        _, mask = cv2.estimateAffinePartial2D(
            ptsA, ptsB,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_thresh,
            maxIters=max_iters,
            confidence=confidence,
        )
        if mask is None:
            return np.zeros((N,), dtype=bool)
        return mask.ravel().astype(bool)[:N]

    return np.ones((N,), dtype=bool)


# ---------- 그리기 ----------
def hstack_images(imA: np.ndarray, imB: np.ndarray, pad: int = 8, color=(30, 30, 30)) -> Tuple[np.ndarray, int]:
    h = max(imA.shape[0], imB.shape[0])
    wA, wB = imA.shape[1], imB.shape[1]
    canvas = np.full((h, wA + pad + wB, 3), color, dtype=np.uint8)
    canvas[:imA.shape[0], :wA] = imA
    canvas[:imB.shape[0], wA + pad:wA + pad + wB] = imB
    return canvas, wA + pad


def draw_matches(
    canvas: np.ndarray,
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    xoffB: int,
    max_lines: int,
    linewidth: int,
    draw_points: bool,
    alpha: int,
    point_radius: int,
) -> None:
    if max_lines <= 0:
        return
    N = min(max_lines, ptsA.shape[0], ptsB.shape[0])
    if N <= 0:
        return

    draw_lines = linewidth > 0
    draw_pts = draw_points and point_radius != 0
    if not draw_lines and not draw_pts:
        return

    overlay = canvas.copy()
    color_line = (255, 200, 0)
    color_ptsA = (0, 220, 255)
    color_ptsB = (80, 255, 80)

    for i in range(N):
        x1, y1 = int(round(ptsA[i, 0])), int(round(ptsA[i, 1]))
        x2, y2 = int(round(ptsB[i, 0] + xoffB)), int(round(ptsB[i, 1]))
        if draw_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), color_line, linewidth, cv2.LINE_AA)
        if draw_pts:
            radius = point_radius if point_radius > 0 else max(1, linewidth + 1)
            cv2.circle(overlay, (x1, y1), radius, color_ptsA, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x2, y2), radius, color_ptsB, -1, cv2.LINE_AA)

    if alpha <= 0:
        canvas[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha / 255.0, canvas, 1.0 - alpha / 255.0, 0, canvas)


# ---------- 메인 ----------
def main():
    # 기본 경로
    root_default = Path(os.environ.get("MATCH_ROOT", str(MATCH_ROOT)))
    out_default = Path(os.environ.get("VIS_ROOT", str(VIS_ROOT)))

    ap = argparse.ArgumentParser(description="Visualize DINOv3 matches (from JSON)")
    ap.add_argument("-r", "--root", type=str, default=str(root_default),
                    help="매칭 JSON 루트 (기본: MATCH_ROOT 또는 /exports/dinov3_match)")
    ap.add_argument("-o", "--out", type=str, default=str(out_default),
                    help="결과 PNG 저장 루트 (기본: VIS_ROOT 또는 /exports/dinov3_vis)")

    # 시각화/RANSAC 옵션 (원본과 동등 동작)
    ap.add_argument("-xl", "--max-lines", type=int, default=int(os.environ.get("VIS_MAX_LINES", 1000)))
    ap.add_argument("-lw", "--linewidth", type=int, default=int(os.environ.get("VIS_LINEWIDTH", 3)))
    ap.add_argument("-al", "--alpha", type=int, default=int(os.environ.get("VIS_ALPHA", 180)))
    point_group = ap.add_mutually_exclusive_group()
    point_group.add_argument("-dp", "--draw-points", dest="draw_points", action="store_true")
    point_group.add_argument("-xp", "--no-points", dest="draw_points", action="store_false")
    ap.set_defaults(draw_points=(os.environ.get("VIS_DRAW_POINTS", "0").strip().lower() in ("1", "true", "yes", "on")))
    ap.add_argument("-pr", "--point-radius", type=int, default=int(os.environ.get("VIS_POINT_RADIUS", -1)))
    ap.add_argument("-R", "--ransac", choices=["off", "affine", "homography"],
                    default=os.environ.get("VIS_RANSAC", "homography").lower())
    ap.add_argument("-m", "--ransac-method", type=str, default=os.environ.get("VIS_RANSAC_IMPL", HOMOGRAPHY_DEFAULT))
    ap.add_argument("-t", "--reproj-th", type=float, default=float(os.environ.get("VIS_REPROJ_TH", 8.0)))
    ap.add_argument("-c", "--confidence", type=float, default=float(os.environ.get("VIS_CONFIDENCE", 0.9999)))
    ap.add_argument("-i", "--iters", type=int, default=int(os.environ.get("VIS_ITERS", 10000)))

    args = ap.parse_args()

    pairs_root = Path(args.root).expanduser()
    if not pairs_root.exists():
        raise SystemExit(f"[error] root 디렉토리가 존재하지 않습니다: {pairs_root}")

    # --- 1차 선택 ---
    top_pick = prompt_pick_top(pairs_root)  # ["**/*.json"] or ["folder/"] or ["file.json"]

    # --- 2차 선택(폴더일 때만) ---
    if len(top_pick) == 1 and top_pick[0].endswith("/"):
        selections = prompt_pick_json_in_folder(pairs_root, top_pick[0])
        if not selections:
            print("[warn] 선택된 항목에서 JSON을 찾지 못했습니다.")
            return
    else:
        selections = top_pick

    json_paths = collect_jsons(pairs_root, selections)
    if not json_paths:
        print("[warn] 선택된 항목에서 JSON을 찾지 못했습니다.")
        return
    print(f"total_json={len(json_paths)}")

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- RANSAC 플래그를 루프 밖에서 한 번만 계산 ---
    # args.ransac_method가 유효 키가 아니면 기본값으로 대체
    ransac_key = args.ransac_method if args.ransac_method in HOMOGRAPHY_METHODS else HOMOGRAPHY_DEFAULT
    homography_flag = HOMOGRAPHY_METHODS.get(ransac_key, cv2.RANSAC)

    # --- 처리 루프: JSON 파싱 → 좌표복원 → RANSAC → 시각화 → PNG 저장 ---
    for jp in json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] JSON 파싱 실패: {jp} ({e})")
            continue

        imgA = Path(data.get("image_a", ""))
        imgB = Path(data.get("image_b", ""))
        if not imgA or not imgB:
            print(f"[skip] image_a/image_b 누락: {jp}")
            continue

        patch = data.get("patch", None)
        if not patch or not patch.get("idx_a") or not patch.get("idx_b"):
            print(f"[skip] no patch matches: {jp}")
            continue

        idx_a = np.array(patch["idx_a"], dtype=np.int64)
        idx_b = np.array(patch["idx_b"], dtype=np.int64)
        n_a = int(patch.get("n_a", len(idx_a)))
        n_b = int(patch.get("n_b", len(idx_b)))

        g_a = patch.get("grid_g_a")
        g_b = patch.get("grid_g_b")

        try:
            imA = np.array(Image.open(str(imgA)).convert("RGB"))[:, :, ::-1]
            imB = np.array(Image.open(str(imgB)).convert("RGB"))[:, :, ::-1]
        except Exception as e:
            print(f"[skip] 이미지 로드 실패: {jp} ({e})")
            continue

        if g_a is not None:
            H_a = W_a = int(g_a)
        else:
            H_a, W_a = best_rect_grid(n_a)
        if g_b is not None:
            H_b = W_b = int(g_b)
        else:
            H_b, W_b = best_rect_grid(n_b)

        ptsA = grid_to_pixels(idx_to_xy(idx_a, H_a, W_a), imA.shape[1], imA.shape[0], W_a, H_a).astype(np.float32)
        ptsB = grid_to_pixels(idx_to_xy(idx_b, H_b, W_b), imB.shape[1], imB.shape[0], W_b, H_b).astype(np.float32)

        mask = ransac_filter(
            ptsA, ptsB,
            args.ransac,
            homography_flag,
            args.reproj_th,
            args.confidence,
            args.iters,
        )
        ptsA_in = ptsA[mask]
        ptsB_in = ptsB[mask]

        canvas, xoffB = hstack_images(imA, imB)
        draw_matches(
            canvas, ptsA_in, ptsB_in, xoffB,
            args.max_lines, args.linewidth, args.draw_points,
            args.alpha, args.point_radius,
        )

        try:
            rel_path = jp.relative_to(pairs_root)
        except ValueError:
            rel_path = Path(jp.name)
        out_path = out_root / rel_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), canvas)
        if ok:
            print(f"[keep] {out_path}")
        else:
            print(f"[warn] 저장 실패: {out_path}")


if __name__ == "__main__":
    main()
