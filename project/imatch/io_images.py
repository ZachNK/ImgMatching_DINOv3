# project/imatch/io_images.py
import re
from pathlib import Path
from typing import Tuple, Dict, List, Iterable
from PIL import Image
import torch
from torchvision import transforms

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
    - a,b 모두 None → 모든 ordered pair (N×(N-1))
    - a만 → (a, k) for k!=a
    - b만 → (k, b) for k!=b
    - 둘 다 있으면 [(a,b)] 한 건
    """
    if a and b:
        return [(a, b)]
    if a:
        return [(a, k) for k in keys if k != a]
    if b:
        return [(k, b) for k in keys if k != b]
    pairs = []
    for i in keys:
        for j in keys:
            if i != j:
                pairs.append((i, j))
    return pairs
