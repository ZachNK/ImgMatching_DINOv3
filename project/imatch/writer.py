# project/imatch/writer.py
import json
from pathlib import Path
from typing import Dict

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
