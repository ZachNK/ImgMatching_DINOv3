# project/imatch/writer.py
import json
from pathlib import Path
from typing import Dict

def save_json(out_dir: Path, stub: str, payload: Dict) -> Path:
    """
    out_dir/stub.json 으로 저장하고 경로 반환
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stub}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path
