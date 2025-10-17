# project/imatch/types.py
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

class RansacMethod(Enum):
    OFF = "off"
    AFFINE = "affine"
    HOMOGRAPHY = "homography"

@dataclass
class MatchConfig:
    image_size: int = 224
    mutual_k: int = 5
    topk: int = 200
    max_patches: int = 0

@dataclass
class RunContext:
    repo_dir: Path
    img_root: Path
    export_dir: Path
    device: str = "cuda"
