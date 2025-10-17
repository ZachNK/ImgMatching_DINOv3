# project/imatch/env.py
import os
from pathlib import Path
from typing import Optional

def getenv(k: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    환경변수 안전 로더. required=True인데 비어있으면 SystemExit.
    """
    v = os.getenv(k, default)
    if required and (v is None or str(v).strip() == ""):
        raise SystemExit(f"Missing env: {k}")
    return v

# 컨테이너 내 기본 경로 (docker-compose.yml/.env 에서 주입)
REPO_DIR   = Path(getenv("REPO_DIR", required=True))
IMG_ROOT   = Path(getenv("IMG_ROOT", required=True))
EXPORT_DIR = Path(getenv("EXPORT_DIR", "/exports/dinov3_embeds"))
PAIR_VIZ_DIR = Path(getenv("PAIR_VIZ_DIR", "/exports/pair_viz"))

# 네트워크 차단 플래그 (torch.hub 원격 다운로드 방지)
DINOV3_BLOCK_NET = getenv("DINOV3_BLOCK_NET", "1").strip() == "1"
