"""Tiny image load/save helpers that degrade gracefully without Pillow."""

from __future__ import annotations

import numpy as np

try:
    from PIL import Image  # type: ignore

    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


def load_gray(path: str) -> np.ndarray:
    """Load an image as a grayscale uint8 array (HxW)."""
    if not _HAVE_PIL:
        raise RuntimeError("Pillow is required to load image files.")
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def save_gray(path: str, image: np.ndarray) -> None:
    """Save a float [0,1] or uint8 grayscale array as an image file."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    if not _HAVE_PIL:
        raise RuntimeError("Pillow is required to save image files.")
    Image.fromarray(arr, mode="L").save(path)
