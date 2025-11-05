# project/imatch/paths.py
"""
Utility helpers for naming pair-match output files.
REPO_DIR: 저장소 디렉터리 경로
IMG_ROOT: 이미지 루트 디렉터리 경로
EMBED_ROOT: 임베드 출력 루트 디렉터리 경로
MATCH_ROOT: 매칭 출력 루트 디렉터리 경로
VIS_ROOT: 시각화 출력 루트 디렉터리 경로
DINOV_BLOCK_NET: 네트워크 차단 설정
JSON: 데이터 키 JSON 파일 경로
IMAGE_KEY: 이미지 키
MODEL_KEY: 모델 키
DATASET_ROOT: 데이터셋 루트 디렉터리 경로
EXPORT_ROOT: 내보내기 루트 디렉터리 경로
img_path(alt: int, img: int) -> List[str]: 이미지 경로 생성
ckpt_path(key: str) -> List[str]: 체크포인트 경로 생성
file_prefix(imgAlt: str, imgIndex: str) -> str: 파일 접두사 생성    
"""
from operator import index
import os
import json
from pathlib import Path
from typing import Optional, List, Dict

# Base directories (docker-compose.yml/.env inject absolute paths)
REPO_DIR = Path(os.getenv("REPO_DIR"))
IMG_ROOT = Path(os.getenv("IMG_ROOT"))
# Output roots (Windows host paths are mounted to /exports inside the container)
EMBED_ROOT = Path(os.getenv("EMBED_ROOT", "/exports/dinov3_embeds"))
MATCH_ROOT = Path(os.getenv("MATCH_ROOT", "/exports/dinov3_match"))
VIS_ROOT = Path(os.getenv("VIS_ROOT", "/exports/dinov3_vis"))
# Network guard: torch.hub remote downloads are disabled unless explicitly opted out
DINOV_BLOCK_NET = os.getenv("DINOV_BLOCK_NET", "1").strip() == "1"
JSON = Path("/workspace/project/json/data_key.json")
with open(JSON, 'r') as s: file = json.load(s)
IMAGE_KEY = file[list(file.keys())[0]]
MODEL_KEY = file[list(file.keys())[1]]
DATASET_ROOT = Path("/opt/datasets")
EXPORT_ROOT = Path("/exports")

def img_path(alt: int, img: int) -> List[str]:
    """
    data_key.json을 참조하여 이미지 데이터 경로명 생성
    e.g. img_path(300, 1) -> ["250912154506_300", "250912154506_300_0001"]
    - 입력:
      alt: int — 이미지 고도
      img: int — 이미지 인덱스
    - 출력:
      List[str] — [폴더 이름, 파일 이름] 형식의 이미지 경로 리스트
    """
    fld = "_".join([[k for k in IMAGE_KEY][9-int(alt/50)], str(alt)])
    dts = "_".join([fld, '%04d'%img])
    result = [fld, dts]
    return result

def ckpt_path(key: str) -> List[str]:
    """
    data_key.json을 참조하여 모델 체크포인트 경로명 생성
    e.g. ckpt_path("vitb16") -> ["dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"]
    - 입력:
      key: str — 모델 키
    - 출력:
      List[str] — [허브 엔트리 이름, 체크포인트 파일 경로] 형식의 리스트
    """
    for p in MODEL_KEY:
        for models in list(MODEL_KEY[p].keys()):
            if key == models:
                folderName = p
                hubEntry = MODEL_KEY[p][models][0]
                fileName = MODEL_KEY[p][models][1]
    result = [hubEntry, "/".join(["/opt", "weights", folderName, fileName])]
    return result

def file_prefix(imgAlt: str, imgIndex: str) -> str:
    """
    파일 접두사 생성
    e.g. file_prefix("300", "1") -> "300_0001"
    - 입력:
      imgAlt: str — 이미지 고도
      imgIndex: str — 이미지 인덱스
    - 출력:
      str — 접두사 문자열
    """
    return f"{imgAlt}_{'%04d'%imgIndex}"



