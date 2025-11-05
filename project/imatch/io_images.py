# project/imatch/io_images.py
import re
from pathlib import Path
from typing import Tuple, Dict, List, Iterable
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms

"""
이미지 파일 I/O 관련 유틸리티 함수들.
"""

def parse_pair(s: str) -> Tuple[int, str]:
    """
    'ALT.FRAME' 형태 파싱: '400.0001' -> (400, '0001')
    """
    alt_s, frm_s = s.split(".", 1)
    alt = int(re.sub(r"\D", "", alt_s))
    frame = re.sub(r"\D", "", frm_s).zfill(4)
    if not frame:
        raise SystemExit("empty frame")
    return alt, frame

def find_image_by_alt_frame(img_root: Path, alt: int, frame: str) -> Path:
    """
    디렉토리 트리에서 '*_{alt}_{frame}.{ext}' 패턴으로 이미지 검색.
    """
    for ext in ("jpg","jpeg","png","bmp","tif","tiff","webp"):
        hits = list(img_root.glob(f"**/*_{alt}_{frame}.{ext}"))
        if hits:
            return hits[0]
    raise SystemExit(f"No image for alt={alt}, frame={frame} under {img_root}")

def load_image_tensor(path: Path) -> torch.Tensor:
    """
    PIL로 RGB 로딩 → ToTensor()
    """
    im = Image.open(path).convert("RGB")
    return transforms.ToTensor()(im)

def scan_images_by_regex(root: Path, regex: str, exts: Iterable[str]) -> Dict[str, Path]:
    """
    정규식에 이름이 매칭되는 이미지 파일을 스캔.
    key='ALT.FRAME' → Path 매핑 반환
    """
    rx = re.compile(regex, re.IGNORECASE)
    exts = tuple(exts)
    out: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue
        m = rx.match(str(p).replace("\\", "/"))
        if not m:
            continue
        alt = m.group("alt")
        frame = m.group("frame")
        key = f"{int(alt)}.{frame}"
        out[key] = p
    if not out:
        raise SystemExit(f"No images matched under {root}")
    return out

def enumerate_pairs(keys: List[str], a: str=None, b: str=None) -> List[Tuple[str,str]]:
    """
    Pair enumeration helper.
    - a, b 모두 None → 모든 ordered pair (N×(N-1))
    - a가 ALT.FRAME → 해당 이미지 vs (b 타깃 또는 전체)
    - a가 ALT → ALT 그룹 전체 vs (b 타깃 또는 전체)
    - b에 대해서도 동일하게 ALT.FRAME / ALT 지원
    """
    key_set = set(keys)
    keys_by_alt: Dict[int, List[str]] = defaultdict(list)
    for key in keys:
        alt_str, frame_str = key.split(".", 1)
        alt_val = int(alt_str)
        keys_by_alt[alt_val].append(key)

    def normalize_target(raw: str, label: str) -> List[str]:
        if raw is None:
            return []
        value = raw.strip()
        if not value:
            return []
        if "." in value:
            alt, frame = parse_pair(value)
            key = f"{alt}.{frame}"
            if key not in key_set:
                raise SystemExit(f"No image matched for {label}={value}")
            return [key]
        # ALT only
        alt_digits = re.sub(r"\D", "", value)
        if not alt_digits:
            raise SystemExit(f"Invalid ALT value for {label}: {value}")
        alt_val = int(alt_digits)
        if alt_val not in keys_by_alt:
            raise SystemExit(f"No images matched ALT={alt_val} for {label}")
        return list(keys_by_alt[alt_val])

    list_a = normalize_target(a, "--pair-a") or list(keys)
    list_b = normalize_target(b, "--pair-b") or list(keys)

    pairs: List[Tuple[str, str]] = []
    for key_a in list_a:
        for key_b in list_b:
            if key_a == key_b:
                continue
            pairs.append((key_a, key_b))
    return pairs
