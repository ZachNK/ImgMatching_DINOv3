"""
DINOv3 torch.hub 모델을 직접 로드해서 단일 이미지를 추론하는 간단한 스크립트.
"""

from __future__ import annotations
import warnings
import os
import math
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from imatch.pretrained import pretrained_model
from imatch.imageprocessing import build_transform
from imatch.extracting import (
    global_embedding,
    patch_embedding,
    patch2grid,
)
from imatch.loading import (
    DATASET_ROOT, 
    EXPORT_ROOT, 
    weights_path, 
    file_prefix, 
    load_image
)


# 백본 모델, 체크포인트 경로, 이미지 경로, 허브 엔트리 이름, 이미지 크기 설정   
# ==== custom ====
# 변경 변수 설정
varAltitude = 300 # 이미지 높이
varIndex = 1 # 이미지 인덱스
varWeight = "vit7b16" # 모델 키
varTargetRes = 1024 # 최대 목표 해상도
"""
"vit7b16", "vitb16", "vith16+", "vitl16", "vits16", "vits16+"
"cxBase", "cxLarge", "cxSmall", "cxTiny"
"vit7b16sat", "vitl16sat" 
"""
# ==== custom ====

HUB_ENTRY = weights_path(varWeight)
WEIGHT_PATH = weights_path(varWeight)[1]
IMG_DIR_NAME = file_prefix(varAltitude, varIndex)

REPO_DIR = Path("/workspace/dinov3")
IMAGE_PATH = DATASET_ROOT / f"{IMG_DIR_NAME}.jpg"
FILE_NAME = f"global_feature_{HUB_ENTRY}_{IMG_DIR_NAME}"

### 메인 함수: 모델 로드, 이미지 전처리, 특징 추출 및 저장

def main() -> None: # 반환값 없음

    ## 1. Prepare Device: torch.device 선택
    # 장치 설정 후 모델 로드: CUDA 사용 가능시 CUDA, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "\n================= Debug: Test patch Embedding =================\n",
        "REPO_DIR: ", REPO_DIR, "\n",
        "IMAGE_PATH: ", IMAGE_PATH, "\n",
        "HUB_ENTRY: ", HUB_ENTRY, "\n",
        "WEIGHT_PATH: ", WEIGHT_PATH, "\n",
        "device: ", device,
        "\n================= Debug: Test patch Embedding =================\n",
    )

    ## 2. Load Model: torch.hub 모델 로드 + 체크포인트 주입
    # DINOv3 모델 로드 후 평가 모드 설정
    model, _ = pretrained_model(REPO_DIR, HUB_ENTRY, WEIGHT_PATH, device)

    ## 3. Load Image: imatch.load_image_tensor로 원본 텐서 확보
    # 이미지 로드 및 전처리
    img_tensor = load_image(IMAGE_PATH.as_posix())

    # 모델의 패치 크기 가져오기
    patch = model.patch_embed.patch_size 

    # 패치 크기에 맞게 이미지 크기 조정 (varTargetRes: 목표 해상도, patch[0]: 패치 크기)
    patch_multiple = math.floor(varTargetRes / patch[0])

    ## 4. Build Preprocess: 패치 크기 기반 transform 빌드, imatch.tfms.build_transform 사용
    # 이미지 전처리 변환 빌드
    transform = build_transform(patch_size=patch[0], patch_multiple=patch_multiple, interpolation="bicubic", normalize=weights_path(varWeight)[2])

    print("img_tensor:", img_tensor.shape)

    # 전처리된 이미지 텐서에 배치 차원 추가 후 장치로 이동
    input_tensor = transform(img_tensor).unsqueeze(0).to(device)

    ### 특징 추출: 글로벌 특징 벡터 & 패치 토큰 추출
    with torch.inference_mode():
        global_vec = global_embedding(model, input_tensor, str(device))
        patch_tokens = patch_embedding(model, input_tensor, str(device))

    # 추출된 텐서 CPU 이동
    global_vec = global_vec.detach().cpu()
    patch_tokens_cpu = None
    patch_grid = None
    if patch_tokens is not None:
        patch_tokens_cpu = patch_tokens.detach().cpu()
        try:
            patch_grid = patch2grid(patch_tokens_cpu)
        except ValueError as err:
            print(f"[warn] patch grid reshape failed: {err}")
    else:
        print("[warn] patch tokens could not be extracted.")

    # 상태 출력
    print("global feature shape:", tuple(global_vec.shape))
    if patch_tokens_cpu is not None:
        print("patch tokens shape:", tuple(patch_tokens_cpu.shape))
        if patch_grid is not None:
            print("patch grid shape:", tuple(patch_grid.shape))

    # 결과 저장 경로 준비
    export_dir = Path("/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    global_npy_path = export_dir / f"{FILE_NAME}.npy"
    global_csv_path = export_dir / f"{FILE_NAME}.csv"
    patch_tokens_path = export_dir / f"patch_tokens_{FILE_NAME}.npy"
    patch_grid_path = export_dir / f"patch_grid_{FILE_NAME}.npy"

    # 글로벌 벡터 저장 (기존 동작 유지)
    global_arr = global_vec.numpy()
    np.save(global_npy_path, global_arr)
    np.savetxt(global_csv_path, global_arr[None, :], delimiter=",")

    # 패치 토큰 및 격자 저장
    if patch_tokens_cpu is not None:
        np.save(patch_tokens_path, patch_tokens_cpu.numpy())
    if patch_grid is not None:
        np.save(patch_grid_path, patch_grid.numpy())

    print(f"[saved] numpy array -> {global_npy_path}")
    print(f"[saved] csv row     -> {global_csv_path}")
    if patch_tokens_cpu is not None:
        print(f"[saved] patch tokens numpy array -> {patch_tokens_path}")
    if patch_grid is not None:
        print(f"[saved] patch grid numpy array   -> {patch_grid_path}")

if __name__ == "__main__":
    main()
