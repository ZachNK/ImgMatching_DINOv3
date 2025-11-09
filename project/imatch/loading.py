# project/imatch/loading.py
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
import re
import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Iterable
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms

# Base directories (docker-compose.yml/.env inject absolute paths)
REPO_DIR = Path(os.getenv("REPO_DIR"))
IMG_ROOT = Path(os.getenv("IMG_ROOT"))
# Output roots (Windows host paths are mounted to /exports inside the container)
EMBED_ROOT = Path(os.getenv("EMBED_ROOT", "/exports/dinov3_embeds"))
MATCH_ROOT = Path(os.getenv("MATCH_ROOT", "/exports/dinov3_match"))
VIS_ROOT = Path(os.getenv("VIS_ROOT", "/exports/dinov3_vis"))
# Network guard: torch.hub remote downloads are disabled unless explicitly opted out

DATASET_ROOT = Path("/opt/datasets")
EXPORT_ROOT = Path("/exports")
WEIGHT_ROOT = Path("/opt/weights")
JSON = Path("/workspace/project/json/data_key.json")

with JSON.open("r", encoding="utf-8") as s:
    registry = json.load(s)

DATASETS: Dict[str, Dict] = registry.get("datasets", {})
WEIGHT_SETS: Dict[str, Dict] = registry.get("weights", {})

def _first_key(data: Dict[str, Dict]) -> str:
    return next(iter(data)) if data else ""

DATASET_KEY = os.getenv("DATASET_KEY", _first_key(DATASETS))
if not DATASET_KEY or DATASET_KEY not in DATASETS:
    raise KeyError(f"[loading] Unknown dataset key: {DATASET_KEY or 'undefined'}")

DATASET_CONFIG = DATASETS[DATASET_KEY]
CAPTURE_MAP = DATASET_CONFIG.get("captures")
if not isinstance(CAPTURE_MAP, dict) or not CAPTURE_MAP:
    raise ValueError(f"[loading] Dataset '{DATASET_KEY}' is missing 'captures' mapping.")

IMAGE_KEY: Dict[str, int] = {str(k): int(v) for k, v in CAPTURE_MAP.items()}
ALTITUDE_TO_CAPTURES: Dict[int, List[str]] = defaultdict(list)
for capture_id, altitude in IMAGE_KEY.items():
    ALTITUDE_TO_CAPTURES[int(altitude)].append(capture_id)

QUERY_CONFIG = DATASET_CONFIG.get("query", {})
QUERY_PREFIX = QUERY_CONFIG.get("prefix", "Q")
QUERY_ROOT = Path(QUERY_CONFIG.get("root", EXPORT_ROOT.as_posix()))

WEIGHTS_KEY = os.getenv("WEIGHTS_KEY", _first_key(WEIGHT_SETS))
if not WEIGHTS_KEY or WEIGHTS_KEY not in WEIGHT_SETS:
    raise KeyError(f"[loading] Unknown weights key: {WEIGHTS_KEY or 'undefined'}")
MODEL_KEY = WEIGHT_SETS[WEIGHTS_KEY]

def _resolve_capture_id(altitude: int) -> str:
    """
    Resolve a capture id from the dataset registry using an altitude value.
    """
    matches = ALTITUDE_TO_CAPTURES.get(int(altitude), [])
    if not matches:
        raise SystemExit(f"[loading(img_path):warn0] No capture mapped to altitude={altitude} for dataset '{DATASET_KEY}'.")
    if len(matches) > 1:
        options = ", ".join(sorted(matches))
        raise SystemExit(f"[loading(img_path):warn0] Ambiguous altitude={altitude}; candidates={options}. Specify DATASET_KEY to disambiguate.")
    return matches[0]


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
    capture_id = _resolve_capture_id(alt)
    folder = f"{capture_id}_{int(alt)}"
    file_name = f"{folder}_{int(img):04d}"
    return [folder, file_name]

def weights_path(key: str) -> List[str]:
    """
    data_key.json을 참조하여 백본모델 경로명 생성
    e.g. weights_path("vits16+") -> ["dinov3_vits16plus", "/opt/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"]
    - 입력:
      key: str — 백본모델 키
    - 출력:
      List[str] — [허브 엔트리 이름, 백본모델 파일 경로] 형식의 리스트
    """
    key = key.strip()
    if not key:
        raise ValueError("[loading(weights_path)] Empty weight key provided.")

    for folder_name, models in MODEL_KEY.items():
        if key in models:
            hub_entry, file_name, data_name = models[key]
            ckpt = WEIGHT_ROOT / folder_name / file_name
            return [hub_entry, ckpt.as_posix(), data_name]
    raise KeyError(f"[loading(weights_path)] Weight key '{key}' not found in registry '{WEIGHTS_KEY}'.")

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
        raise SystemExit("[loading(parse_pair):warn1] empty frame")
    return alt, frame

def find_image(img_root: Path, alt: int, frame: str) -> Path:
    """
    디렉토리 트리에서 '*_{alt}_{frame}.{ext}' 패턴으로 이미지 검색.
    """
    for ext in ("jpg","jpeg","png","bmp","tif","tiff","webp"):
        hits = list(img_root.glob(f"**/*_{alt}_{frame}.{ext}"))
        if hits:
            return hits[0]
    raise SystemExit(f"[loading(find_image):warn2] No image for alt={alt}, frame={frame} under {img_root}")

def load_image(path: Path) -> torch.Tensor:
    """
    PIL로 RGB 로딩 → ToTensor()
    """
    im = Image.open(path).convert("RGB")
    return transforms.ToTensor()(im)

def images_regex(root: Path, regex: str, exts: Iterable[str]) -> Dict[str, Path]:
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
        raise SystemExit(f"[loading:warn3] No images matched under {root}")
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
                raise SystemExit(f"[loading:warn3] No image matched for {label}={value}")
            return [key]
        # ALT only
        alt_digits = re.sub(r"\D", "", value)
        if not alt_digits:
            raise SystemExit(f"[loading:warn3] Invalid ALT value for {label}: {value}")
        alt_val = int(alt_digits)
        if alt_val not in keys_by_alt:
            raise SystemExit(f"[loading:warn3] No images matched ALT={alt_val} for {label}")
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

def save_json(out_dir: Path, stub: str, payload: Dict) -> Path:
    """
    out_dir/stub.json 으로 저장하고 경로 반환
    - 입력: 
      out_dir: Path — 출력 디렉터리 경로
      stub: str — 파일 이름 접두사
      payload: Dict — JSON으로 저장할 데이터 딕셔너리
    - 출력:
      Path — 저장된 JSON 파일 경로
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stub}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def prepare_run_context(args, weight_groups: Dict[str, List[str]], all_weight_keys: List[str]):
    """이미지 목록과 가중치 경로를 준비한다."""
    key2path = images_regex(IMG_ROOT, args.regex, args.exts)
    keys = sorted(key2path.keys(), key=lambda s: (int(s.split('.')[0]), s.split('.')[1]))
    pairs = enumerate_pairs(keys, args.pair_a, args.pair_b)
    print(f"[images] total={len(keys)}  pairs_to_run={len(pairs)}")

    selected_weight_keys = (
        all_weight_keys if args.all_weights
        else weight_groups[args.group] if args.group
        else args.weights
    )
    all_weight_key_set = set(all_weight_keys)
    resolved_weights = []
    for weight_name in selected_weight_keys:
        if weight_name not in all_weight_key_set:
            raise SystemExit(f"Unknown weight key: {weight_name}")
        hub_entry, weight_path_str, dataset_type = weights_path(weight_name)
        weight_path = Path(weight_path_str)
        if not weight_path.is_file():
            raise SystemExit(f"[weight] not found for {weight_name}: {weight_path}")
        resolved_weights.append((weight_name, hub_entry, weight_path))
    print(f"[weights] selected={len(resolved_weights)} -> {[w[0] for w in resolved_weights]}")
    return key2path, pairs, resolved_weights, dataset_type


def save_match_result(
    args,
    weight_name: str,
    hub_entry: str,
    weight_path: Path,
    pair_a: str,
    pair_b: str,
    image_a: Path,
    image_b: Path,
    cosine: float,
    time_ms: Dict[str, float],
    patch: Dict | None,
) -> Path:
    """매칭 결과를 JSON으로 저장하고 경로 반환."""
    model_root = f"/opt/weights/{weight_path.name}"
    meta = dict(
        repo_dir=str(REPO_DIR),
        img_root=str(IMG_ROOT),
        embed_root=str(EMBED_ROOT),
        match_root=str(MATCH_ROOT),
        model_root=model_root,
        hub_model=hub_entry,
        device=args.device,
        image_size=int(args.image_size),
    )
    payload = dict(
        meta=meta,
        image_a=str(image_a),
        image_b=str(image_b),
        weight=weight_name,
        cosine=cosine,
        time_ms=dict(
            forward_a=float(time_ms["forward_a"]),
            forward_b=float(time_ms["forward_b"]),
            total=float(time_ms["total"]),
        ),
    )
    payload["advanced_settings"] = dict(
        match_threshold=float(args.match_th),
        max_features=int(args.max_features),
        keypoint_threshold=float(args.keypoint_th),
        line_threshold=float(args.line_th),
        matching_mode="mutual_knn_k1_unique",
    )
    if patch is not None:
        payload["patch"] = patch

    alt_id, frame_id = pair_a.split('.')
    out_dir = MATCH_ROOT / f"{weight_name}_{alt_id}_{frame_id}"
    out_name = f"{weight_name}_{pair_a}_{pair_b}"
    out_path = save_json(out_dir, out_name, payload)
    print(f"[saved] {out_path}")
    return out_path
