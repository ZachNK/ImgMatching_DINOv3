# project/imatch/ckpt_finder.py
"""
체크포인트 탐색 유틸리티.

주의:
- 최신 구조에서는 YAML(config)에서 alias→파일명을 직접 관리하는 방식을 권장.
- 본 모듈은 레거시/보조 용도로, 지정한 루트들에서 glob 패턴을 모아 탐색합니다.
- Adapters 관련 필터는 완전히 제거되었습니다.
"""
from pathlib import Path
from typing import List

# (선택) 레거시 alias → glob 패턴 매핑 (필요 시만 사용)
ALIASES_GLOB = {
    "vit7b16":    "*vit7b16*.pth",
    "vitb16":     "*vitb16*.pth",
    "vith16+":    "*vith16plus*.pth",
    "vitl16":     "*vitl16*.pth",
    "vits16":     "*vits16*.pth",
    "vits16+":    "*vits16plus*.pth",
    "cxBase":     "*convnext*base*.pth",
    "cxLarge":    "*convnext*large*.pth",
    "cxSmall":    "*convnext*small*.pth",
    "cxTiny":     "*convnext*tiny*.pth",
    "vit7b16sat": "*vit7b16*sat*.pth",
    "vitl16sat":  "*vitl16*sat*.pth",
}

def collect_ckpts(roots: List[Path], patterns: List[str]) -> List[Path]:
    """
    여러 루트에서 glob 패턴으로 .pth 파일들을 수집.
    """
    hits: List[Path] = []
    for r in roots:
        for pat in patterns:
            hits.extend(sorted(r.glob(f"**/{pat}")))
    # 중복 제거
    seen = set()
    out: List[Path] = []
    for p in hits:
        if p in seen:
            continue
        out.append(p)
        seen.add(p)
    return out
