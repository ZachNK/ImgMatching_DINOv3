from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from PIL import Image

MAX_SELECTIONS = 100
FIGURE_DPI = 100


def prompt_existing_path(message: str) -> Path:
    """Interactively request a path from the user until an existing file is provided."""
    while True:
        raw = input(message).strip()
        if not raw:
            print("Please enter a path.")
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print(f"File not found: {path}")


def prompt_int(message: str, min_value: int, max_value: int) -> int:
    """Prompt for an integer within [min_value, max_value]."""
    while True:
        raw = input(message).strip()
        if not raw:
            print("Please enter a value.")
            continue
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if not (min_value <= value <= max_value):
            print(f"Enter an integer between {min_value} and {max_value}.")
            continue
        return value


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert float image data to uint8 while preserving relative structure for display."""
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if np.isclose(max_val, min_val):
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_visual(path: Path) -> Tuple[np.ndarray, dict]:
    """Load an image or numpy array for visualization and return data plus plotting kwargs."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
        if data.ndim == 2:
            return data, {"cmap": "viridis", "aspect": "equal"}
        if data.ndim == 3:
            if data.shape[-1] in (3, 4):
                return normalize_to_uint8(data), {"aspect": "equal"}
            if data.shape[0] in (3, 4):
                transposed = np.moveaxis(data, 0, -1)
                return normalize_to_uint8(transposed), {"aspect": "equal"}
        raise ValueError(f"Unsupported array shape for visualization: {path} (shape={data.shape})")

    try:
        with Image.open(path) as im:
            return np.array(im), {"aspect": "equal"}
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to open image: {path}") from exc


def get_hw(array: np.ndarray) -> Tuple[int, int]:
    """Return height and width for plotting."""
    if array.ndim == 2:
        return array.shape[0], array.shape[1]
    if array.ndim == 3:
        return array.shape[0], array.shape[1]
    raise ValueError(f"Unsupported array dimensions: {array.shape}")


def build_figure_layout(
    original: np.ndarray,
    original_kwargs: dict,
    comparisons: List[np.ndarray],
    comparison_kwargs: List[dict],
    padding_percent: int,
    titles: List[str],
) -> None:
    """Construct and display the matplotlib figure according to the requested layout."""
    orig_h, orig_w = get_hw(original)
    if comparisons:
        comp_dims = [get_hw(arr) for arr in comparisons]
        comp_total_width = sum(w for _, w in comp_dims)
        comp_max_height = max(h for h, _ in comp_dims)
        width_ratios = [w for _, w in comp_dims]
    else:
        comp_total_width = 0
        comp_max_height = 1
        width_ratios = [orig_w]

    total_width = max(orig_w, comp_total_width if comparisons else 0, 1)
    total_height = orig_h + (comp_max_height if comparisons else 0)

    figsize = (total_width / FIGURE_DPI, total_height / FIGURE_DPI if total_height > 0 else 5)
    fig = plt.figure(figsize=figsize, dpi=FIGURE_DPI)

    if comparisons:
        gs = fig.add_gridspec(
            2,
            len(comparisons),
            height_ratios=[orig_h, comp_max_height],
            width_ratios=width_ratios,
        )
        ax_main = fig.add_subplot(gs[0, :])
    else:
        gs = fig.add_gridspec(1, 1)
        ax_main = fig.add_subplot(gs[0])

    main_kwargs = dict(original_kwargs)
    if original.ndim == 2 and "cmap" not in main_kwargs:
        main_kwargs["cmap"] = "viridis"
    ax_main.imshow(original, **main_kwargs)
    ax_main.set_title(titles[0])
    ax_main.axis("off")

    for idx, (data, kwargs, title) in enumerate(zip(comparisons, comparison_kwargs, titles[1:]), start=0):
        ax = fig.add_subplot(gs[1, idx] if comparisons else gs[0])
        plot_kwargs = dict(kwargs)
        if data.ndim == 2 and "cmap" not in plot_kwargs:
            plot_kwargs["cmap"] = "viridis"
        ax.imshow(data, **plot_kwargs)
        ax.set_title(title)
        ax.axis("off")

    pad_fraction = max(0.0, min(padding_percent, 100)) / 100.0
    margin = min(pad_fraction * 0.5, 0.49)
    fig.subplots_adjust(
        left=margin,
        right=1 - margin,
        top=1 - margin,
        bottom=margin,
        wspace=0.05,
        hspace=0.2,
    )

    save_raw = input("Enter a path to save the composed figure (press Enter to skip): ").strip()
    if save_raw:
        save_path = Path(save_raw).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.show()


def main() -> None:
    print("=== Feature Map Comparison Viewer ===")
    print("Select the original image and up to 100 feature-map images to compare.")

    original_path = prompt_existing_path("Path to the original image: ")
    compare_count = prompt_int(
        f"Number of comparison images (0-{MAX_SELECTIONS}): ",
        min_value=0,
        max_value=MAX_SELECTIONS,
    )

    comparison_paths: List[Path] = []
    for idx in range(1, compare_count + 1):
        path = prompt_existing_path(f"Path for comparison image {idx}: ")
        comparison_paths.append(path)

    padding = prompt_int("Padding level (0-100, percent of canvas reserved as margin): ", 0, 100)

    try:
        original_data, original_kwargs = load_visual(original_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to load the original image: {exc}")
        sys.exit(1)

    comparison_data: List[np.ndarray] = []
    comparison_kwargs: List[dict] = []
    titles: List[str] = [original_path.name]
    for path in comparison_paths:
        try:
            data, kwargs = load_visual(path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to load comparison image ({path}): {exc}")
            sys.exit(1)
        comparison_data.append(data)
        comparison_kwargs.append(kwargs)
        titles.append(path.name)

    build_figure_layout(
        original=original_data,
        original_kwargs=original_kwargs,
        comparisons=comparison_data,
        comparison_kwargs=comparison_kwargs,
        padding_percent=padding,
        titles=titles,
    )


if __name__ == "__main__":
    main()
