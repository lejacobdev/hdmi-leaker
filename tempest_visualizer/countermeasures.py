"""
Countermeasures: the defensive half of TEMPEST.

Understanding leakage is only useful if you can reduce it. This module
demonstrates two classic, well-documented mitigations and lets you *measure*
their effect with the same reconstruction pipeline used elsewhere.

  * TEMPEST-safe fonts (Kuhn & Anderson, 1998): removing the top of the spatial
    spectrum from on-screen text drastically cuts the high-frequency transitions
    that radiate, at a modest cost in sharpness. We approximate this with a
    low-pass filter on the image.

  * Reduced refresh / lower pixel clock: moving the emanation energy and
    lowering transition rates. (Represented at the timing level, not here.)

These are for evaluating your own setup. The metric functions let you quantify
"how much does the ghost image resemble the original" before and after a
mitigation.
"""

from __future__ import annotations

import numpy as np


def lowpass_image(image: np.ndarray, cutoff: float = 0.25) -> np.ndarray:
    """Apply an isotropic low-pass filter in the frequency domain.

    Parameters
    ----------
    image:
        2-D grayscale array (any range); returned in the same range as input.
    cutoff:
        Normalised radial cutoff in (0, 1]. Lower = softer = less leakage.

    This is a didactic stand-in for a properly designed TEMPEST font: it shows
    that suppressing high spatial frequencies suppresses the reconstructable
    signal.
    """
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy**2 + fx**2) / (0.5 * np.sqrt(2))
    mask = np.exp(-(radius**2) / (2 * max(cutoff, 1e-3) ** 2))
    out = np.fft.ifft2(np.fft.fft2(arr) * mask).real
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation between two images in [-1, 1].

    A practical proxy for "how legible is the reconstruction". Images are
    resized (nearest) to a common shape before comparison.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        b = _resize_nn(b, *a.shape)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def leakage_score(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """0..1 score: how recognisable the reconstruction is (higher = worse leak).

    Emanations reveal *edges*, and a reconstruction has an unknown horizontal
    roll and may include blanking. So we compare edge maps on a common small
    grid and search a few circular shifts, taking the best alignment. This makes
    the score reflect legibility rather than incidental misalignment.
    """
    a = _edges(_resize_nn(np.asarray(original, float), 96, 128))
    b = _edges(_resize_nn(np.abs(np.asarray(reconstructed, float)), 96, 128))
    a = a - a.mean()
    best = 0.0
    for sx in range(-6, 7, 2):
        for sy in range(-4, 5, 2):
            bb = np.roll(np.roll(b, sy, axis=0), sx, axis=1)
            bb = bb - bb.mean()
            denom = np.sqrt((a**2).sum() * (bb**2).sum())
            if denom <= 0:
                continue
            best = max(best, abs(float((a * bb).sum() / denom)))
    return float(np.clip(best, 0.0, 1.0))


def _edges(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    return gx + gy


def _resize_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    ys = np.clip((np.arange(out_h) * in_h / out_h).astype(np.int64), 0, in_h - 1)
    xs = np.clip((np.arange(out_w) * in_w / out_w).astype(np.int64), 0, in_w - 1)
    return arr[ys][:, xs]
