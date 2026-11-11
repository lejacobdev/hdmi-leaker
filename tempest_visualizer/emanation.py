"""
Forward model: turn an image into a simulated electromagnetic emanation.

The physics, simplified
------------------------
A digital video link (DVI/HDMI) serialises each pixel at the pixel clock. The
cable and connectors act as an (unintended) antenna. What radiates is *not* the
DC luminance of each pixel but the high-frequency content — the *transitions*
between consecutive symbols. That is why a leaked frame, once reconstructed,
looks like a ghostly edge map rather than a faithful photograph: flat areas
radiate almost nothing, sharp edges and text radiate strongly.

A receiver tuned near a harmonic of the pixel clock sees a complex baseband
signal that is, to first order:

    s[n]  ~=  HPF( pixel_stream )[n] * exp(j * (phase + 2*pi*df*n))  +  noise

where ``HPF`` is a high-pass / differencing operation (transition emphasis),
``df`` a small residual carrier offset, and ``noise`` additive Gaussian noise
whose level sets the capture SNR.

This module implements exactly that, in a transparent, inspectable way. It is a
teaching model, not a calibrated RF simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timing import DisplayTiming


@dataclass
class EmanationParams:
    """Knobs for the forward model."""

    samples_per_pixel: int = 1
    """Receiver oversampling relative to the pixel clock."""

    snr_db: float = 10.0
    """Signal-to-noise ratio of the simulated capture, in dB."""

    carrier_phase: float = 0.0
    """Fixed carrier phase offset (radians)."""

    freq_offset_cycles_per_pixel: float = 0.0
    """Residual carrier frequency offset, in cycles per pixel sample."""

    transition_emphasis: float = 1.0
    """Blend between pure transition emphasis (1.0) and raw amplitude (0.0).

    Real emanations are dominated by transitions; lowering this is only useful
    for illustrating *why* edges dominate.
    """

    phase_noise_std: float = 0.0
    """Std-dev of per-sample phase jitter (radians), modelling oscillator drift."""

    noise_sigma: float | None = None
    """Absolute noise std-dev per I/Q component. Overrides ``snr_db`` when set.

    This models a *fixed* ambient/thermal noise floor that does not shrink when
    the on-screen content radiates less. It is what makes countermeasure
    comparisons physically meaningful: filter the image, the signal weakens, the
    noise floor stays put, and the reconstruction degrades.
    """

    seed: int | None = 0
    """RNG seed for reproducibility. ``None`` for nondeterministic noise."""


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """Coerce an HxW or HxWxC uint8/float image to float32 in [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        # Rec. 601 luma
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def build_raster(image: np.ndarray, timing: DisplayTiming) -> np.ndarray:
    """Place the active image into a full raster (with blanking) of the mode.

    The image is resized by nearest-neighbour to ``(v_active, h_active)`` if it
    does not already match, then padded with zeros to ``(v_total, h_total)``.
    Returns a float32 array in [0, 1].
    """
    gray = _to_gray_float(image)
    h, w = gray.shape
    if (h, w) != (timing.v_active, timing.h_active):
        gray = _resize_nn(gray, timing.v_active, timing.h_active)
    raster = np.zeros((timing.v_total, timing.h_total), dtype=np.float32)
    raster[: timing.v_active, : timing.h_active] = gray
    return raster


def _resize_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    ys = (np.arange(out_h) * in_h / out_h).astype(np.int64)
    xs = (np.arange(out_w) * in_w / out_w).astype(np.int64)
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)
    return arr[ys][:, xs]


def emphasised_stream(
    image: np.ndarray, timing: DisplayTiming, params: EmanationParams
) -> np.ndarray:
    """Return the real-valued transition-emphasised, oversampled scan stream.

    This is the clean (noise-free, unmodulated) magnitude envelope the receiver
    would see. Exposed so callers can measure signal power and derive a fixed
    absolute noise floor for fair before/after comparisons.
    """
    raster = build_raster(image, timing)
    stream = raster.reshape(-1)
    diff = np.diff(stream, prepend=stream[:1])
    emphasised = (
        params.transition_emphasis * diff
        + (1.0 - params.transition_emphasis) * stream
    )
    spp = max(1, int(params.samples_per_pixel))
    if spp > 1:
        emphasised = np.repeat(emphasised, spp)
    return emphasised.astype(np.float32)


def signal_power(image: np.ndarray, timing: DisplayTiming, params: EmanationParams) -> float:
    """Mean power of the emanation envelope for ``image`` under ``params``."""
    e = emphasised_stream(image, timing, params)
    return float(np.mean(np.abs(e) ** 2)) + 1e-12


def sigma_for_snr(reference_power: float, snr_db: float) -> float:
    """Per-component noise std-dev giving ``snr_db`` against ``reference_power``."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    return float(np.sqrt((reference_power / snr_lin) / 2.0))


def simulate_emanation(
    image: np.ndarray,
    timing: DisplayTiming,
    params: EmanationParams | None = None,
) -> np.ndarray:
    """Simulate the complex baseband emanation for one frame.

    Parameters
    ----------
    image:
        Source image (grayscale or RGB). Represents what is on screen.
    timing:
        The display mode driving the (virtual) cable.
    params:
        Forward-model parameters. Defaults to :class:`EmanationParams`.

    Returns
    -------
    np.ndarray
        Complex64 1-D array of length ``v_total * h_total * samples_per_pixel``
        in scan order (row-major), i.e. the IQ stream a receiver would digitise.
    """
    params = params or EmanationParams()
    rng = np.random.default_rng(params.seed)

    emphasised = emphasised_stream(image, timing, params)
    n = emphasised.shape[0]

    # Carrier phase, residual frequency offset, and phase noise.
    phase = params.carrier_phase + 2.0 * np.pi * params.freq_offset_cycles_per_pixel * np.arange(n)
    if params.phase_noise_std > 0:
        phase = phase + np.cumsum(rng.normal(0.0, params.phase_noise_std, size=n))
    baseband = emphasised * np.exp(1j * phase).astype(np.complex64)

    # Noise: either a fixed absolute floor (noise_sigma) or one derived from the
    # per-frame SNR. The absolute floor is what makes countermeasures measurable.
    if params.noise_sigma is not None:
        sigma = float(params.noise_sigma)
    else:
        sig_power = float(np.mean(np.abs(baseband) ** 2)) + 1e-12
        sigma = sigma_for_snr(sig_power, params.snr_db)
    noise = (
        rng.normal(0.0, sigma, size=n) + 1j * rng.normal(0.0, sigma, size=n)
    ).astype(np.complex64)

    return (baseband + noise).astype(np.complex64)
