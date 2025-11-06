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
        task = progress.add_task("      Processing...", total=None)

        result = run(*args, **kwargs)
        progress.update(task, completed=1)
    return result

def token_preview(tokens):
    def _flatten(seq):
        for item in seq:
            if isinstance(item, (list, tuple)):
                yield from _flatten(item)
            else:
                yield item

    raw = tokens.tolist() if hasattr(tokens, "tolist") else tokens
    values = list(_flatten(raw)) if isinstance(raw, (list, tuple)) else [raw]
    length = len(values)

    if length == 0:
        return "[]"
    if length >= 6:
        head_count = tail_count = 3
    elif length >= 4:
        head_count = tail_count = 2
    elif length >= 2:
        head_count = tail_count = 1
    else:
        head_count, tail_count = 1, 0

    if head_count + tail_count >= length:
        display_values = values
    else:
        display_values = values[:head_count]
        display_values += ["..."] + values[-tail_count:] if tail_count else ["..."]
    return "[" + ", ".join(str(item) for item in display_values) + "]"

