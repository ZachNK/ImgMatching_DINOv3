"""
Generate rotated + center-cropped query images for rotation-robustness tests.

The script relies on the helpers in imatch.querycreating so future automation
can import and reuse the same logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from imatch.querycreating import generate_queries_for_directory


ANGLES: Sequence[float] = (45.0, 90.0, 135.0, 180.0)
CROP_RATIO: float = 0.5


@dataclass(frozen=True)
class QueryTask:
    source: Path
    destination: Path


TASKS: Iterable[QueryTask] = (
    QueryTask(
        source=Path("/opt/datasets/250912150549_400"),
        destination=Path("/exports/Q250912150549_400"),
    ),
    QueryTask(
        source=Path("/opt/datasets/250912154506_300"),
        destination=Path("/exports/Q250912154506_300"),
    ),
    QueryTask(
        source=Path("/opt/datasets/250912161658_200"),
        destination=Path("/exports/Q250912161658_200"),
    ),
)


def main() -> None:
    total_outputs = 0
    for task in TASKS:
        src = task.source.expanduser()
        dst = task.destination.expanduser()

        if not src.exists():
            print(f"[WARN] Source directory missing, skipping: {src}")
            continue

        print(f"[INFO] Generating queries for {src} -> {dst}")
        results = generate_queries_for_directory(
            source_dir=src,
            destination_dir=dst,
            angles=ANGLES,
            crop_ratio=CROP_RATIO,
            overwrite=True,
            resize_to_original=False,
        )
        total_outputs += len(results)
        print(f"\tGenerated {len(results)} query images.")

    print(f"[DONE] Total query images generated: {total_outputs}")


if __name__ == "__main__":
    main()
