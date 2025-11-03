"""Compute DINOv3 patch-level cosine similarity maps and store them as NumPy files."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict

from imatch.features import extract_patch_tokens, reshape_patch_tokens_to_grid
from imatch.io_images import load_image_tensor
from imatch.tfms import build_transform

# ==== custom ====
varH = 300
varI = 1
varKEY = "vits16"
varSIZE = 1024

""" >>>> Weight Key: HUB_ENTRY
"vit7b16": "dinov3_vit7b16",
"vitb16": "dinov3_vitb16",
"vith16+": "dinov3_vith16plus",
"vitl16": "dinov3_vitl16",
"vits16": "dinov3_vits16",
"vits16+": "dinov3_vits16plus",

"cxBase": "convnext_base",
"cxLarge": "convnext_large",
"cxSmall": "convnext_small",
"cxTiny": "convnext_tiny",

"vit7b16sat": "dinov3_vit7b16",
"vitl16sat": "dinov3_vitl16",
"""
# ==== custom ====

# ==== DECODING ====
JSON = Path("/workspace/project/json/data_key.json")
with open(JSON, 'r') as s: file = json.load(s)
IMAGE_KEY = file[list(file.keys())[0]]
MODEL_KEY = file[list(file.keys())[1]]

def img_path(alt: int, img: int) -> str:
    fld = "_".join([[k for k in IMAGE_KEY][9-int(alt/50)], str(alt)])
    dts = "_".join([fld, '%04d'%img])
    result = "/".join([fld, dts])
    return result

def ckpt_path(key: str) -> List[str]:
    for p in MODEL_KEY:
        for models in list(MODEL_KEY[p].keys()):
            if key == models:
                folderName = p
                hubEntry = MODEL_KEY[p][models][0]
                fileName = MODEL_KEY[p][models][1]                
    result = [hubEntry, "/".join(["/opt","weights", folderName, fileName])]
    return result

IMG_DIR_NAME = img_path(varH, varI)
HUB_ENTRY = ckpt_path(varKEY)[0]
CKPT_PATH = ckpt_path(varKEY)[1]
lst = IMG_DIR_NAME.split("/")[-1].split("_")
if len(lst) < 3:
    raise ValueError(f"Expected at least three underscore-separated tokens in {IMG_DIR_NAME!r}.")
REPO_DIR = Path("/workspace/dinov3")
DATASET_ROOT = Path("/opt/datasets")
EXPORT_ROOT = Path("/exports")
IMAGE_PATH = DATASET_ROOT / f"{IMG_DIR_NAME}.jpg"
FILE_NAME = f"global_feature_{HUB_ENTRY}_{lst[1]}_{lst[2]}"
FILE_PREFIX = f"{HUB_ENTRY}_{lst[1]}_{lst[2]}"
# ==== DECODING ====

# ====DEBUGING=======
# print("REPO_DIR: ", REPO_DIR, "HUB_ENTRY: ", HUB_ENTRY, "CKPT_PATH: ", CKPT_PATH)

# ====DEBUGING=======

def load_dinov3_model(repo_dir: str, hub_entry: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    torch.hub에서 DINOv3 모델을 불러오고, 체크포인트를 로드해 키를 정리한 뒤 지정한 장치(GPU/CPU)로 올림.

    입력: 
    repo_dir: Path(DINOv3 모델 경로), hub_entry: str(백본 모델명), ckpt_path: Path(백본 모델 파일), device: torch.device(CPU/GPU 대상).
    
    출력: 
    torch.nn.Module를 반환하며, 누락/예상치 못한 키를 로그로 출력.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    model = torch.hub.load(repo_dir.as_posix(), hub_entry, source="local", trust_repo=True, pretrained=False)

    map_location = device if device.type == "cpu" else torch.device(device.type, device.index or 0)
    try:
        state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[ckpt][warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[ckpt][warn] unexpected keys: {len(unexpected)}")

    return model.to(device).eval()


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
    cosine_path = EXPORT_ROOT / f"patch_cosine_map_{FILE_PREFIX}.npy"
    np.save(cosine_path, cosine_map)
    print(f"[saved] cosine map -> {cosine_path}")

    if grid_hw is not None:
        reshaped = cosine_map.reshape(grid_hw[0], grid_hw[1], grid_hw[0], grid_hw[1])
        reshaped_path = EXPORT_ROOT / f"patch_cosine_grid_{FILE_PREFIX}.npy"
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
    img_tensor = load_image_tensor(image_path.as_posix())

    patch_size = get_patch_size(model)
    patch_multiple = max(1, math.floor(image_size / patch_size))
    transform = build_transform(
        patch_size=patch_size,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=True,
    )

    input_tensor = transform(img_tensor).unsqueeze(0).to(device)
    with torch.inference_mode():
        tokens = extract_patch_tokens(model, input_tensor, str(device))

    if tokens is None:
        raise RuntimeError("Patch tokens could not be extracted from the model output.")

    tokens = tokens.detach()

    try:
        tokens_grid = reshape_patch_tokens_to_grid(tokens)
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
        "REPO_DIR: ", REPO_DIR, "\n", 
        "IMAGE_PATH: ", IMAGE_PATH, "\n",
        "HUB_ENTRY: ", HUB_ENTRY, "\n", 
        "CKPT_PATH: ", CKPT_PATH, "\n", 
        "device: ", device, 
        "\n================= Debug: Test Feature Map =================\n"
        )

    model = load_dinov3_model(REPO_DIR, HUB_ENTRY, CKPT_PATH, device)

    generate_patch_cosine(
        model=model,
        image_path=IMAGE_PATH,
        image_size=varSIZE,
        device=device,
    )


if __name__ == "__main__":
    main()
