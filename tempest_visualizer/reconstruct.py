"""
Reconstruction: recover a ghost frame from a complex IQ stream.

Given an IQ capture (simulated here, or recorded from your own equipment) the
reconstruction problem is:

  1. Find the horizontal period — how many samples make up one scan line. This
     is the strongest periodicity in the signal and is found by autocorrelation.
  2. Find the vertical period — how many lines make up one frame. Same trick,
     on the line-energy signal.
  3. Fold the 1-D magnitude stream into a 2-D image of that width and height.
  4. Align (de-skew) so the picture does not "roll", exactly like tuning an old
     analogue TV's hold controls.

No knowledge of the source image is used. The only prior is "video is a
periodic raster", which is what makes TEMPEST reconstruction possible — and
what countermeasures try to break.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReconstructResult:
    image: np.ndarray            # 2-D float array in [0, 1], the ghost frame
    line_length: int             # detected samples per line
    frame_height: int            # detected lines per frame
    confidence: float            # 0..1 strength of the detected periodicity


def _autocorr_peak(
    x: np.ndarray, min_lag: int, max_lag: int
) -> tuple[int, float]:
    """Return (lag, normalised_strength) of the strongest autocorrelation peak.

    Uses FFT-based autocorrelation for speed. ``strength`` is the peak height
    relative to the zero-lag energy, a rough 0..1 confidence.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    if n < 4 or max_lag <= min_lag:
        return min_lag, 0.0
    max_lag = min(max_lag, n - 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    zero = ac[0] if ac[0] != 0 else 1.0
    window = ac[min_lag : max_lag + 1]
    if window.size == 0:
        return min_lag, 0.0
    k = int(np.argmax(window))
    lag = min_lag + k
    strength = float(np.clip(window[k] / zero, 0.0, 1.0))
    return lag, strength


def detect_line_length(
    iq: np.ndarray, min_len: int = 100, max_len: int = 6000
) -> tuple[int, float]:
    """Estimate samples-per-line from the IQ magnitude via autocorrelation."""
    mag = np.abs(np.asarray(iq))
    return _autocorr_peak(mag, min_len, max_len)


def detect_frame_height(
    iq: np.ndarray, line_length: int, min_lines: int = 400, max_lines: int = 4000
) -> tuple[int, float]:
    """Estimate lines-per-frame from per-line energy.

    ``min_lines`` defaults to 400 because real display modes have tall rasters
    (v_total >= ~500). Allowing very short periods lets text-row periodicity win
    the autocorrelation and report a spurious sub-harmonic frame height.
    """
    mag = np.abs(np.asarray(iq))
    n_lines = mag.shape[0] // line_length
    if n_lines < 4:
        return max(1, n_lines), 0.0
    line_energy = mag[: n_lines * line_length].reshape(n_lines, line_length).sum(axis=1)
    lo = min(min_lines, max(4, n_lines // 2))
    return _autocorr_peak(line_energy, lo, min(max_lines, n_lines - 1))


def _fold(mag: np.ndarray, width: int, height: int) -> np.ndarray:
    needed = width * height
    if mag.shape[0] < needed:
        mag = np.pad(mag, (0, needed - mag.shape[0]))
    return mag[:needed].reshape(height, width)


def _deskew(frame: np.ndarray) -> np.ndarray:
    """Remove constant horizontal roll by aligning each row's energy centroid.

    A real capture has an unknown horizontal start offset that makes the image
    wrap. We estimate a single global shift from the column-energy profile and
    roll the whole frame, which is enough to stop the picture from splitting.
    """
    col_energy = frame.sum(axis=0)
    if col_energy.sum() <= 0:
        return frame
    # Shift so the darkest column (blanking) sits at the right edge.
    shift = int(np.argmin(col_energy))
    return np.roll(frame, -shift, axis=1)


def _normalise(frame: np.ndarray) -> np.ndarray:
    lo = np.percentile(frame, 1.0)
    hi = np.percentile(frame, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((frame - lo) / (hi - lo), 0.0, 1.0)


def reconstruct_frame(
    iq: np.ndarray,
    line_length: int | None = None,
    frame_height: int | None = None,
    deskew: bool = True,
) -> ReconstructResult:
    """Reconstruct a ghost frame from an IQ stream.

    If ``line_length`` / ``frame_height`` are given, they override the
    autocorrelation detector (useful when you already know the display mode).
    """
    iq = np.asarray(iq)
    mag = np.abs(iq).astype(np.float64)

    conf_h = conf_v = 1.0
    if line_length is None:
        line_length, conf_h = detect_line_length(iq)
    if frame_height is None:
        frame_height, conf_v = detect_frame_height(iq, line_length)

    line_length = max(1, int(line_length))
    frame_height = max(1, int(frame_height))

    frame = _fold(mag, line_length, frame_height)
    if deskew:
        frame = _deskew(frame)
    frame = _normalise(frame)

    return ReconstructResult(
        image=frame,
        line_length=line_length,
        frame_height=frame_height,
        confidence=float(np.sqrt(max(conf_h, 0.0) * max(conf_v, 0.0))),
    )
