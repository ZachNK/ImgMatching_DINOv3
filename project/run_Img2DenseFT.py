"""
Batch runner that sequentially executes Test_global_embedding and
Generate_dense_feature for every combination of altitude, index, and weight.
"""

from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.Generate_DenseFT import generate_dense_feature
from project.Test_Embedding import run_test_global_embedding


ALTITUDES: Sequence[int] = (100, 150, 200, 250, 300, 350, 400, 450)
INDICES: Sequence[int] = tuple(range(1, 290))  # 1 ~ 289 inclusive
WEIGHTS: Sequence[str] = (
    "vit7b16",
    "vitb16",
    "vith16+",
    "vitl16",
    "vits16",
    "vits16+",
    "cxBase",
    "cxLarge",
    "cxSmall",
    "cxTiny",
    "vit7b16sat",
    "vitl16sat",
)


def iter_tasks(
    altitudes: Sequence[int],
    indices: Sequence[int],
    weights: Sequence[str],
) -> Iterable[Tuple[int, int, str]]:
    """Yield every combination in the requested order."""
    return itertools.product(altitudes, indices, weights)


def run_all(
    altitudes: Sequence[int] = ALTITUDES,
    indices: Sequence[int] = INDICES,
    weights: Sequence[str] = WEIGHTS,
) -> List[Tuple[int, int, str, str]]:
    """Run both pipelines for each combination, collecting failures."""
    total = len(altitudes) * len(indices) * len(weights)
    tasks = iter_tasks(altitudes, indices, weights)
    failures: List[Tuple[int, int, str, str]] = []

    for idx, (altitude, index_val, weight) in enumerate(tasks, start=1):
        print(
            f"[{idx}/{total}] altitude={altitude}, index={index_val}, weight={weight}"
        )
        try:
            run_test_global_embedding(
                altitude=altitude,
                index=index_val,
                weight=weight,
            )
            generate_dense_feature(
                altitude=altitude,
                index=index_val,
                weight=weight,
            )
        except Exception:
            failure = traceback.format_exc()
            failures.append((altitude, index_val, weight, failure))
            print(
                f"[warn] Failed for altitude={altitude}, index={index_val}, weight={weight}"
            )
            print(failure)

    if failures:
        print("\n=== Summary of failures ===")
        for altitude, index_val, weight, failure in failures:
            print(f"* altitude={altitude}, index={index_val}, weight={weight}")
            print(failure)
    else:
        print("\nAll combinations processed successfully.")

    return failures


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
