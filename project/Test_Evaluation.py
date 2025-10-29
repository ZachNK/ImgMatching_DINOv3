import os, sys, argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List

## 입력 정의
FILE_ROOT = Path("/exports")

def _list_folder(folder: Path) -> List[str]:
    """폴더 내 npy파일을 재귀로 수집 → 폴더 기준 상대경로 문자열 리스트 반환"""
    if not folder.exists():
        return []

    rels = set()
    for npy_file in folder.rglob("*.npy"):
        try:
            rels.add(str(npy_file.relative_to(folder)))
        except Exception:
            rels.add(npy_file.name)

    return sorted(rels)

def pick_npy(root: Path) -> List[str]:
    """루트 이하 npy 목록을 보여주고 선택한 순번의 경로를 반환"""
    npy_list = _list_folder(root)

    if not npy_list:
        print("[info] npy 파일을 찾지 못했습니다.")
        return []

    print("npy 파일 목록:")
    for idx, rel in enumerate(npy_list, start=1):
        print(f"  {idx:2d}. {rel}")

    choice = input(f"선택할 번호 입력 (1~{len(npy_list)}, Enter = 취소): ").strip()

    if not choice:
        return []
    if not choice.isdigit():
        print("[warn] 잘못된 입력. 작업 종료.")
        return []

    ndx = int(choice)
    if not 1 <= ndx <= len(npy_list):
        print("[warn] 번호 범위 초과. 작업 종료.")
        return []

    selected = npy_list[ndx - 1]
    return [str(root / selected)]

def load_file(paths: List[str]) -> np.ndarray:
    if not paths:
        raise ValueError("[error] 경로가 비어 있습니다.")
    npy = np.load(paths[0])
    return npy


def validate_embedding(embedding: np.ndarray) -> None:
    """임베딩 기본 검증: 차원 수, NaN/Inf 여부, dtype 확인"""
    if embedding.ndim not in {1, 2}:
        raise ValueError(f"[error] 지원하지 않는 임베딩 형태: ndim={embedding.ndim}")
    if not np.issubdtype(embedding.dtype, np.floating):
        raise ValueError(f"[error] 부동소수 dtype 필요. 현재 dtype={embedding.dtype}")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("[error] NaN/Inf 값을 포함한 임베딩입니다.")


def compute_basic_stats(embedding: np.ndarray) -> Dict[str, float]:
    """평균, 표준편차, 최소/최대, L2 노름 평균·표준편차 계산"""
    arr = embedding
    if arr.ndim == 1:
        arr = arr[None, :]

    norms = np.linalg.norm(arr, axis=1)
    stats: Dict[str, float] = {
        "dim": float(arr.shape[1]),
        "num_samples": float(arr.shape[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
    }
    return stats


def compute_similarity_stats(embedding_a: np.ndarray, embedding_b: np.ndarray) -> Dict[str, float]:
    """두 임베딩 간 코사인 유사도 분포 계산"""
    arr_a = embedding_a
    arr_b = embedding_b

    if arr_a.ndim == 1:
        arr_a = arr_a[None, :]
    if arr_b.ndim == 1:
        arr_b = arr_b[None, :]

    if arr_a.shape[1] != arr_b.shape[1]:
        raise ValueError(f"[error] 차원이 일치하지 않습니다: {arr_a.shape[1]} vs {arr_b.shape[1]}")
    if arr_a.shape[0] != arr_b.shape[0]:
        raise ValueError(f"[error] 샘플 수가 다릅니다: {arr_a.shape[0]} vs {arr_b.shape[0]}")

    denom = np.linalg.norm(arr_a, axis=1) * np.linalg.norm(arr_b, axis=1)
    denom = np.clip(denom, 1e-12, None)
    cosine = np.einsum("ij,ij->i", arr_a, arr_b) / denom

    stats: Dict[str, float] = {
        "num_pairs": float(cosine.shape[0]),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std()),
        "cosine_min": float(cosine.min()),
        "cosine_max": float(cosine.max()),
    }
    return stats

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluation Tokens")
    ap.add_argument("-r", "--root", type=str, default=str(FILE_ROOT))
    args = ap.parse_args()

    exports_root = Path(args.root).expanduser()
    if not exports_root.exists():
        raise SystemExit(f"[error] /exports 디렉토리가 존재하지 않습니다. {exports_root}")
    
    ## 데이터 적재 계층
    selection1 = pick_npy(exports_root)
    if not selection1:
        print("[info] 첫 번째 선택이 없어 종료합니다.")
        return
    file_npy_1 = load_file(selection1)
    print(f"[info] 선택1: {selection1[0]} -> shape={file_npy_1.shape}")

    selection2 = pick_npy(exports_root)
    if not selection2:
        print("[info] 두 번째 선택이 없어 종료합니다.")
        return
    file_npy_2 = load_file(selection2)
    print(f"[info] 선택2: {selection2[0]} -> shape={file_npy_2.shape}")
    

    ## 기초 검증 단계
    validate_embedding(file_npy_1)
    validate_embedding(file_npy_2)

    ## 통계 계산 모듈
    stats1 = compute_basic_stats(file_npy_1)
    stats2 = compute_basic_stats(file_npy_2)
    sim_stats = compute_similarity_stats(file_npy_1, file_npy_2)

    print("[stats] 첫 번째 임베딩")
    for key, value in stats1.items():
        print(f"  {key:>12}: {value}")

    print("[stats] 두 번째 임베딩")
    for key, value in stats2.items():
        print(f"  {key:>10}: {value}")

    print("[similarity] 코사인 유사도")
    for key, value in sim_stats.items():
        print(f"  {key:>12}: {value}")
    ## 결과 저장/보고




    return None


if __name__ == "__main__":
    main()
