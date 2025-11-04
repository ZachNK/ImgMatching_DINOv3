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
from imatch.features import (
    extract_global_feature,
    extract_patch_tokens,
    reshape_patch_tokens_to_grid,
)
from imatch.paths import img_path, ckpt_path, file_prefix, DATASET_ROOT, EXPORT_ROOT
from imatch.models import load_model
from imatch.io_images import load_image_tensor
from imatch.tfms import build_transform

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

HUB_ENTRY = ckpt_path(varWeight)[0]
CKPT_PATH = ckpt_path(varWeight)[1]
IMG_DIR_NAME = img_path(varAltitude, varIndex)
REPO_DIR = Path("/workspace/dinov3")
IMAGE_PATH = DATASET_ROOT / f"{IMG_DIR_NAME[0]}/{IMG_DIR_NAME[1]}.jpg"
FILE_NAME = f"global_feature_{HUB_ENTRY}_{file_prefix(varAltitude, varIndex)}"


### 메인 함수: 모델 로드, 이미지 전처리, 특징 추출 및 저장

def main() -> None: # 반환값 없음
    ### 특징 벡터(배열)를 numpy 배열과 CSV로 저장
    ### 0. Export Features: numpy 와 csv로 배열 저장
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    npy_path = EXPORT_ROOT / f"{FILE_NAME}.npy"
    csv_path = EXPORT_ROOT / f"{FILE_NAME}.csv"
    grid_path = EXPORT_ROOT / f"patch_grid_{FILE_NAME}.npy"
    ## 1. Prepare Device: torch.device 선택
    # 장치 설정 후 모델 로드: CUDA 사용 가능시 CUDA, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "\n================= Debug: Test Global Embedding =================\n",
        f"REPO_DIR: {REPO_DIR}\n",
        f"IMAGE_PATH: {IMAGE_PATH}\n",
        f"HUB_ENTRY: {HUB_ENTRY}\n",
        f"CKPT_PATH: {CKPT_PATH}\n",
        f"device: {device}\n",
        f"Test Global embedding DINOv3 numpy array -> {npy_path}\n",
        f"Test Global embedding DINOv3 csv row     -> {csv_path}\n",
        f"Test Global patch grid numpy array       -> {grid_path}\n",
        "\n================= Debug: Test Global Embedding =================\n",
    )

    ## 2. Load Model: torch.hub 모델 로드 + 체크포인트 주입
    # DINOv3 모델 로드 후 평가 모드 설정
    model, _ = load_model(REPO_DIR, HUB_ENTRY, CKPT_PATH, device)

    ## 3. Load Image: imatch.load_image_tensor로 원본 텐서 확보
    # 이미지 로드 및 전처리
    img_tensor = load_image_tensor(IMAGE_PATH.as_posix())

    # 모델의 패치 크기 가져오기
    patch = model.patch_embed.patch_size 

    # 패치 크기에 맞게 이미지 크기 조정 (varTargetRes: 목표 해상도, patch[0]: 패치 크기)
    patch_multiple = math.floor(varTargetRes / patch[0])

    ## 4. Build Preprocess: 패치 크기 기반 transform 빌드, imatch.tfms.build_transform 사용
    # 이미지 전처리 변환 빌드
    transform = build_transform(patch_size=patch[0], patch_multiple=patch_multiple, interpolation="bicubic", normalize=True)

    print("img_tensor:", img_tensor.shape)

    # 전처리된 이미지 텐서에 배치 차원 추가 후 장치로 이동
    input_tensor = transform(img_tensor).unsqueeze(0).to(device)

    ### 특징 추출: 글로벌 특징 벡터 & 패치 토큰 추출
    ## 5. Run Inference: transform 적용, imatch.features.extract_global_feature 사용
    with torch.inference_mode():
        # global_vec: 추출된 글로벌 특징 벡터
        global_vec = extract_global_feature(model, input_tensor, str(device))
        patch_tokens = extract_patch_tokens(model, input_tensor, str(device))

    ### 결과 출력 및 저장
    # ※※※global_vec: 특징 벡터 ※※※
    # CPU로 이동 후 그래프 분리
    global_vec = global_vec.detach().cpu()
    patch_grid = None
    if patch_tokens is not None:
        patch_tokens = patch_tokens.detach().cpu()
        try:
            patch_grid = reshape_patch_tokens_to_grid(patch_tokens)
        except ValueError as err:
            print(f"[warn] patch grid reshape failed: {err}")
    else:
        print("[warn] patch tokens could not be extracted.")

    # 상태 출력
    print("Global feature shape:", tuple(global_vec.shape))
    # 값 출력 (리스트 형태)
    print("Global feature:", global_vec.tolist())

    if patch_tokens is not None:
        print("Patch tokens shape:", tuple(patch_tokens.shape))
        if patch_grid is not None:
            print("Patch grid shape:", tuple(patch_grid.shape))


    

    # numpy로 저장
    global_arr = global_vec.numpy()
    np.save(npy_path, global_arr)
    # csv로 저장
    np.savetxt(csv_path, global_arr[None, :], delimiter=",")

    # 저장완료메세지 출력
    print(f"[saved] Test Global embedding DINOv3 numpy array -> {npy_path}")
    print(f"[saved] Test Global embedding DINOv3 csv row     -> {csv_path}")
    if patch_grid is not None:
        np.save(grid_path, patch_grid.numpy())
        print(f"[saved] Test Global patch grid numpy array           -> {grid_path}")

if __name__ == "__main__":
    main()
