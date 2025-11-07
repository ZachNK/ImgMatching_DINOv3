"""Compute DINOv3 patch-level cosine similarity maps and store them as NumPy files."""
from __future__ import annotations
import warnings
import os
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
from imatch.pretrained import pretrained_model
from imatch.preprocess import build_transform
from imatch.extracting import (
    global_embedding,
    patch_embedding,
    patch2grid,
)
from imatch.loading import (
    DATASET_ROOT, 
    EXPORT_ROOT, 
    img_path, 
    weights_path, 
    file_prefix, 
    load_image

)


# 백본 모델, 체크포인트 경로, 이미지 경로, 허브 엔트리 이름, 이미지 크기 설정   
# ==== custom ====
# 변경 변수 설정
varAltitude = 400 # 이미지 높이
varIndex = 100 # 이미지 인덱스
varWeight = "vits16+" # 모델 키
varTargetRes = 1024 # 최대 목표 해상도
"""
"vit7b16", "vitb16", "vith16+", "vitl16", "vits16", "vits16+"
"cxBase", "cxLarge", "cxSmall", "cxTiny"
"vit7b16sat", "vitl16sat" 
"""
# ==== custom ====

HUB_ENTRY = weights_path(varWeight)[0]
CKPT_PATH = weights_path(varWeight)[1]
IMG_DIR_NAME = img_path(varAltitude, varIndex)

REPO_DIR = Path("/workspace/dinov3")
IMAGE_PATH = DATASET_ROOT / f"{IMG_DIR_NAME[0]}/{IMG_DIR_NAME[1]}.jpg"
FILE_NAME = f"FTM_{HUB_ENTRY}_{file_prefix(varAltitude, varIndex)}"

def get_patch_size(model: torch.nn.Module) -> int:
    """
    DINOv3모델이 제공하는 patch_embed.patch_size를 읽어 패치 한 변의 길이를 정수로 반환.

    입력: 
    model: torch.nn.Module

    출력:
    int 하나(패치 크기). 필요한 속성이 없으면 AttributeError를 발생시킴.
    """
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is None or not hasattr(patch_embed, "patch_size"):
        raise AttributeError("Model does not expose patch_embed.patch_size")
    size = patch_embed.patch_size
    return int(size[0]) if isinstance(size, (tuple, list)) else int(size)


def compute_cosine_map(tokens: torch.Tensor) -> np.ndarray:
    tokens = tokens.to(dtype=torch.float32, device="cpu")
    tokens = F.normalize(tokens, p=2, dim=1)
    cosine = tokens @ tokens.t()
    return cosine.numpy()


def export_outputs(
    cosine_map: np.ndarray,
    grid_hw: tuple[int, int] | None,
) -> None:
    """
    계산된 코사인 맵과 (있다면) 격자 정보를 받아 출력 디렉터리를 만들고 두 종류의 .npy 파일을 생성하는 저장 담당 함수.

    입력:
    cosine_map: np.ndarray — 패치 코사인 유사도 행렬.
    grid_hw: tuple[int, int] | None — 패치 격자 형태(행, 열). 정사각형일 때 (h, w)로 전달되고, 알 수 없으면 None.

    출력:
    반환값은 None. 대신 부수효과로 /exports 아래에 결과 파일을 저장합니다.
    항상 patch_cosine_map_{FILE_PREFIX}.npy 파일을 저장.
    grid_hw가 주어지면 4차원 텐서로 재구성한 patch_cosine_grid_{FILE_PREFIX}.npy도 추가 저장.
    """
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    cosine_path = EXPORT_ROOT / f"FTM_patch_cosine_map_{FILE_NAME}.npy"
    np.save(cosine_path, cosine_map)
    print(f"[saved] cosine map -> {cosine_path}")

    if grid_hw is not None:
        reshaped = cosine_map.reshape(grid_hw[0], grid_hw[1], grid_hw[0], grid_hw[1])
        reshaped_path = EXPORT_ROOT / f"FTM_patch_cosine_grid_{FILE_NAME}.npy"
        np.save(reshaped_path, reshaped)
        print(f"[saved] cosine grid -> {reshaped_path}")


def generate_patch_cosine(
    model: torch.nn.Module,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> None:
    """
    generate_patch_cosine() 함수는 DINOv3 모델로 단일 이미지를 처리해 패치 임베딩의 코사인 유사도 맵을 만들어 저장하는 한 사이클을 맡음. 
    project/feature_map.py (lines 92-128)에서 이미지를 텐서로 불러와 패치 크기에 맞춘 변환을 적용.
    모델에서 패치 토큰을 추출한 뒤 정규화하여 코사인 유사도 행렬을 계산. 이후 격자 크기를 구해 간단한 로그를 찍고, 
    export_outputs()를 통해 /exports/patch_cosine_map_{…}.npy와 /exports/patch_cosine_grid_{…}.npy 파일을 생성.

    입력:
    model: torch.nn.Module — DINOv3 모델.
    image_path: Path — 처리할 이미지 경로.
    image_size: int — 입력 이미지가 리사이즈될 목표 크기 (패치 변환용)
    device: torch.device — 모델과 텐서를 올릴 장치 (GPU/CPU)

    출력:
    반환값은 None; 대신 다음 작업을 수행:
    - 이미지를 로드하고 변환해 패치 토큰을 추출.
    - 코사인 유사도 행렬을 계산하고, 가능한 경우 격자 크기를 산출.
    - export_outputs()을 호출해 /exports 폴더에 patch_cosine_map_{FILE_PREFIX}.npy와 (격자 정보가 있다면), patch_cosine_grid_{FILE_PREFIX}.npy를 저장.

    """
    img_tensor = load_image(image_path.as_posix())

    patch_size = get_patch_size(model)
    patch_multiple = max(1, math.floor(image_size / patch_size))
    transform = build_transform(
        patch_size=patch_size,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=varWeight,
    )

    input_tensor = transform(img_tensor).unsqueeze(0).to(device)
    with torch.inference_mode():
        tokens = patch_embedding(model, input_tensor, str(device))

    if tokens is None:
        raise RuntimeError("Patch tokens could not be extracted from the model output.")

    tokens = tokens.detach()

    try:
        tokens_grid = patch2grid(tokens)
        grid_hw = tokens_grid.shape[:2]
    except ValueError as err:
        tokens_grid = None
        grid_hw = None
        print(f"[warn] token grid reshape failed: {err}")

    cosine_map = compute_cosine_map(tokens)

    grid_info = "x".join(map(str, grid_hw)) if grid_hw else "unknown"
    print(f"[info] image={image_path}")
    print(f"       tokens={tokens.shape}  cosine={cosine_map.shape}  grid={grid_info}")

    ## 출력
    export_outputs(
        cosine_map=cosine_map,
        grid_hw=grid_hw,
    )


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "================= Debug: Test Feature Map =================\n", 
        f"REPO_DIR: {REPO_DIR}\n", 
        f"IMAGE_PATH: {IMAGE_PATH}\n",
        f"HUB_ENTRY: {HUB_ENTRY}\n", 
        f"CKPT_PATH: {CKPT_PATH}\n", 
        f"device: {device}\n",
        f"export directory: {EXPORT_ROOT}/FTM_cosine_grid_{FILE_NAME}.npy\n"
        f"export directory: {EXPORT_ROOT}/FTM_cosine_map_{FILE_NAME}.npy\n"
        "================= Debug: Test Feature Map =================\n"
        )

    model, _ = pretrained_model(REPO_DIR, HUB_ENTRY, CKPT_PATH, device)

    generate_patch_cosine(
        model=model,
        image_path=IMAGE_PATH,
        image_size=varTargetRes,
        device=device,
    )


if __name__ == "__main__":
    main()
