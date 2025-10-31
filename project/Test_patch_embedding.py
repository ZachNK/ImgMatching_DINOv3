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
from imatch.features import extract_global_feature
from imatch.io_images import load_image_tensor
from imatch.tfms import build_transform

# 백본 모델, 체크포인트 경로, 이미지 경로, 허브 엔트리 이름, 이미지 크기 설정   
# ==== custom ====
IMG_DIR_NAME = "250912154506_300/250912154506_300_0001"
CKPT_PATH = P("/opt/weights/01_ViT_LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
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
        state = torch.load(CKPT_PATH.as_posix(), map_location="cuda:0")
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
    # 장치 설정 및 모델 로드: CUDA 사용 가능 시 CUDA, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 2. Loaf Model: torch.hub 모델 로드 + 체크포인트 주입
    # DINOv3 모델 로드 및 평가 모드 설정
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

    # 전처리된 이미지 텐서를 배치 차원 추가 후 장치로 이동
    input_tensor = transform(img_tensor).unsqueeze(0).to(device)

    ### 특징 추출: 전역 특징 벡터 추출
    ## 5. Run Inference: transform 적용, imatch.features.extract_global_feature 사용
    with torch.inference_mode():
        # global_vec: 추출된 전역 특징 벡터
        global_vec = extract_global_feature(model, input_tensor, str(device))

    ### 결과 출력 및 저장
    # ※※※ global_vec: 특징 벡터 ※※※
    # CPU로 이동 및 그래프 분리
    global_vec = global_vec.detach().cpu()
    
    # 형태 출력
    print("global feature shape:", tuple(global_vec.shape))
    # 값 출력 (리스트 형태)
    print("global feature:", global_vec.tolist())

    ### 특징 벡터(임베딩)을 numpy 배열 및 CSV로 저장 → 6. Export Features: numpy 및 csv로 임베딩 저장
    export_dir = P("/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    npy_path = export_dir / f"{FILE_NAME}.npy"
    csv_path = export_dir / f"{FILE_NAME}.csv"

    # numpy로 저장
    global_arr = global_vec.numpy()
    np.save(npy_path, global_arr)
    # csv로 저장
    np.savetxt(csv_path, global_arr[None, :], delimiter=",")
    
    # 저장완료 메세지 출력
    print(f"[saved] numpy array -> {npy_path}")
    print(f"[saved] csv row     -> {csv_path}")


if __name__ == "__main__":
    main()
