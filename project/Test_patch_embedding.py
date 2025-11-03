"""
DINOv3 torch.hub 모델을 직접 로드해서 단일 이미지를 추론하는 간단한 스크립트.
"""

from __future__ import annotations
import os
import warnings
import math
import json
import torch
import numpy as np
from pathlib import Path as P
from typing import List, Dict
from imatch.features import (
    extract_global_feature,
    extract_patch_tokens,
    reshape_patch_tokens_to_grid,
)
from imatch.io_images import load_image_tensor
from imatch.tfms import build_transform

# 백본 모델, 체크포인트 경로, 이미지 경로, 허브 엔트리 이름, 이미지 크기 설정   
# ==== custom ====
IMG_DIR_NAME = "250912154506_300/250912154506_300_0001"
CKPT_PATH = P("/opt/weights/01_ViT_LVD-1689M/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
IMAGE_SIZE = 1024
# ==== custom ====

lst = IMG_DIR_NAME.split("/")[-1].split("_")
REPO_DIR = P("/workspace/dinov3")
IMAGE_PATH = P(f"/opt/datasets/{IMG_DIR_NAME}.jpg")
HUB_ENTRY = "_".join(os.path.splitext(os.path.basename(CKPT_PATH))[0].split("_")[:2])
FILE_NAME = f"patch_feature_{HUB_ENTRY}_{lst[1]}_{lst[2]}"

# DINOv3 모델 로드 함수
def load_dinov3_model() -> torch.nn.Module: # torch.nn.Module 반환
    warnings.filterwarnings("ignore", category=UserWarning)
    ### 로컬에서 DINOv3 백본 모델 로드
    # REPO_DIR 경로에서 허브 엔트리 이름으로 모델 로드, 사전 학습된 가중치 없이 로드
    # HUB_ENTRY: 허브에서 로드할 모델의 이름
    # source="local": 로컬 경로에서 모델을 로드
    # trust_repo=True: 신뢰할 수 있는 저장소로 간주 (보안 경고 비활성화)

    model = torch.hub.load(REPO_DIR.as_posix(), HUB_ENTRY, source="local", trust_repo=True, pretrained=False)
    
    ### 체크포인트 로드 및 모델 가중치 설정
    try:
        # CUDA 장치에 맞게 체크포인트 로드 시도
        state = torch.load(CKPT_PATH.as_posix(), map_location="cuda:0", weights_only=True)
    except TypeError:
        # 실패 시 CPU에 맞게 로드
        state = torch.load(str(CKPT_PATH), map_location="cpu")
    # 체크포인트에서 'state_dict' 키가 있으면 해당 값으로 설정
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # 'module.' 접두사가 있는 키를 제거하여 모델에 맞게 정리
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # 모델에 가중치 로드, 엄격하지 않게 설정하여 누락된 키나 예기치 않은 키 경고 출력
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        # 누락된 키 경고 출력
        print(f"[ckpt][warn] missing keys: {len(missing)}")
    if unexpected:
        # 예기치 않은 키 경고 출력
        print(f"[ckpt][warn] unexpected keys: {len(unexpected)}")
    # 모델 반환    
    return model

### 메인 함수: 모델 로드, 이미지 전처리, 특징 추출 및 저장

def main() -> None: # 반환값 없음

    ## 1. Prepare Device: torch.device 선택
    # 장치 설정 후 모델 로드: CUDA 사용 가능시 CUDA, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 2. Loaf Model: torch.hub 모델 로드 + 체크포인트 주입
    # DINOv3 모델 로드 후 평가 모드 설정
    model = load_dinov3_model().to(device).eval()

    ## 3. Load Image: imatch.load_image_tensor로 원본 텐서 확보
    # 이미지 로드 및 전처리
    img_tensor = load_image_tensor(IMAGE_PATH.as_posix())

    # 모델의 패치 크기 가져오기
    patch = model.patch_embed.patch_size 

    # 이미지 크기
    desired_size = IMAGE_SIZE

    # 패치 크기에 맞게 이미지 크기 조정 (desired_size: 이미지크기, patch[0]: 패치 크기)
    patch_multiple = math.floor(desired_size / patch[0])

    ## 4. Build Preprocess: 패치 크기 기반 transform 빌드, imatch.tfms.build_transform 사용
    # 이미지 전처리 변환 빌드
    transform = build_transform(patch_size=patch[0], patch_multiple=patch_multiple, interpolation="bicubic", normalize=True)

    print("img_tensor:", img_tensor.shape)

    # 전처리된 이미지 텐서에 배치 차원 추가 후 장치로 이동
    input_tensor = transform(img_tensor).unsqueeze(0).to(device)

    ### 특징 추출: 글로벌 특징 벡터 & 패치 토큰 추출
    with torch.inference_mode():
        global_vec = extract_global_feature(model, input_tensor, str(device))
        patch_tokens = extract_patch_tokens(model, input_tensor, str(device))

    # 추출된 텐서 CPU 이동
    global_vec = global_vec.detach().cpu()
    patch_tokens_cpu = None
    patch_grid = None
    if patch_tokens is not None:
        patch_tokens_cpu = patch_tokens.detach().cpu()
        try:
            patch_grid = reshape_patch_tokens_to_grid(patch_tokens_cpu)
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
    export_dir = P("/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    global_npy_path = export_dir / f"{FILE_NAME}.npy"
    global_csv_path = export_dir / f"{FILE_NAME}.csv"
    patch_tokens_path = export_dir / f"{FILE_NAME}_tokens.npy"
    patch_grid_path = export_dir / f"{FILE_NAME}_grid.npy"

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
