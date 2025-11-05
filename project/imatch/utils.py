# project/imatch/cli_utils.py
"""
Helpers for building CLI argument parsers.
"""

import argparse
from collections.abc import Callable
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from typing import Callable

# e.g. bounded_float(low=0.0, high=1.0) -> 
def bounded_float(low: float, high: float) -> Callable[[str], float]:
    """
    Return an argparse type validator enforcing low <= value <= high.
    """
    def _validate(raw: str) -> float:
        value = float(raw)
        if not (low <= value <= high):
            raise argparse.ArgumentTypeError(f"value {value} not in [{low}, {high}]")
        return value
    return _validate


def bounded_int(low: int, high: int) -> Callable[[str], int]:
    """
    Return an argparse type validator enforcing low <= value <= high.
    """
    def _validate(raw: str) -> int:
        value = int(raw)
        if not (low <= value <= high):
            raise argparse.ArgumentTypeError(f"value {value} not in [{low}, {high}]")
        return value
    return _validate



def progress_bar(run: Callable, *args, **kwargs):
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), TimeElapsedColumn()) as progress:
        task = progress.add_task("Processing...", total=None)

        result = run(*args, **kwargs)
        progress.update(task, completed=1)
    return result

