# project/imatch/paths.py
"""
Utility helpers for naming pair-match output files.
"""
from operator import index
import os
import json
from pathlib import Path
from typing import Optional, List, Dict

# def getenv(key: str, default: Optional[str] = None, required: bool = False) -> str:
#     """
#     Environment loader with optional required enforcement.
#     """
#     value = os.getenv(key, default)
#     if required and (value is None or str(value).strip() == ""):
#         raise SystemExit(f"Missing env: {key}")
#     return value


# def split_key(key: str) -> tuple[str, str]:
#     alt, frame = key.split(".")
#     return alt, frame


# def out_dir_for_pair(weight_alias: str, key_a: str) -> Path:
#     alt, frame = split_key(key_a)
#     return MATCH_ROOT / f"{weight_alias}_{alt}_{frame}"


# def out_name_for_pair(weight_alias: str, key_a: str, key_b: str) -> str:
#     return f"{weight_alias}_{key_a}_{key_b}"

# def model_path(image_height: int, image_index: int, model_key: str, ) -> Path:
#     """
#     Generate the model checkpoint path based on image height, index, and model key.
#     """
#     folder_name = f"model_{image_height}_{image_index}"
#     file_name = f"{model_key}_checkpoint.pth"
#     return MATCH_ROOT / "models" / folder_name / file_name

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
    fld = "_".join([[k for k in IMAGE_KEY][9-int(alt/50)], str(alt)])
    dts = "_".join([fld, '%04d'%img])
    result = [fld, dts]
    return result

def ckpt_path(key: str) -> List[str]:
    for p in MODEL_KEY:
        for models in list(MODEL_KEY[p].keys()):
            if key == models:
                folderName = p
                hubEntry = MODEL_KEY[p][models][0]
                fileName = MODEL_KEY[p][models][1]
    result = [hubEntry, "/".join(["/opt", "weights", folderName, fileName])]
    return result

def file_prefix(imgAlt: str, imgIndex: str) -> str:
    return f"{imgAlt}_{imgIndex}"



