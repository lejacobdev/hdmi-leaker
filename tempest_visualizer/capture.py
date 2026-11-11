"""
Optional live capture via SoapySDR — for authorized self-assessment only.

This module is intentionally minimal and generic. It samples a block of IQ from
a SoapySDR-supported radio into the same complex64 array the rest of the toolkit
uses. It performs **no** targeting, scanning, or device fingerprinting, and it
prints an authorization reminder before doing anything.

Intended use: measuring the emanations of equipment **you own or are explicitly
authorized to test**, e.g. verifying that a shielding change actually reduced
leakage. Capturing signals radiated by other people's equipment is unlawful in
most places. You are responsible for operating within the law and within the
terms of any authorization you hold.

SoapySDR is an optional dependency; if it is not installed, importing this
module still works but :func:`capture_block` raises a clear error.
"""

from __future__ import annotations

import numpy as np

_AUTH_NOTICE = (
    "[authorization] Live SDR capture is for equipment you own or are "
    "explicitly authorized to test. Intercepting third-party emanations is "
    "illegal in most jurisdictions."
)


def capture_block(
    sample_rate: float,
    center_freq: float,
    num_samples: int,
    *,
    gain_db: float | None = None,
    device_args: str = "",
    confirm_authorized: bool = False,
) -> np.ndarray:
    """Capture ``num_samples`` of IQ from a SoapySDR device.

    Parameters
    ----------
    sample_rate, center_freq:
        Receiver sample rate and tuning frequency, in Hz.
    num_samples:
        Number of complex samples to read.
    gain_db:
        Optional overall gain; ``None`` uses the driver default.
    device_args:
        SoapySDR device selection string, e.g. ``"driver=rtlsdr"``.
    confirm_authorized:
        Must be set ``True`` by the caller to acknowledge the authorization
        notice. Prevents casual/accidental use.
    """
    if not confirm_authorized:
        raise PermissionError(
            "Refusing to capture: pass confirm_authorized=True to acknowledge "
            "that you own or are authorized to test the source equipment.\n"
            + _AUTH_NOTICE
        )
    print(_AUTH_NOTICE)

    try:
        import SoapySDR  # type: ignore
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32  # type: ignore
    except Exception as exc:  # pragma: no cover - hardware/optional dep
        raise RuntimeError(
            "SoapySDR is not available. Install it (and your SDR driver) to use "
            "live capture, or work from recorded IQ files instead."
        ) from exc

    sdr = SoapySDR.Device(device_args)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, float(sample_rate))
    sdr.setFrequency(SOAPY_SDR_RX, 0, float(center_freq))
    if gain_db is not None:
        sdr.setGain(SOAPY_SDR_RX, 0, float(gain_db))

    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(stream)
    try:
        out = np.empty(num_samples, dtype=np.complex64)
        filled = 0
        buff = np.empty(1 << 16, dtype=np.complex64)
        while filled < num_samples:
            sr = sdr.readStream(stream, [buff], buff.size)
            n = sr.ret
            if n <= 0:
                continue
            take = min(n, num_samples - filled)
            out[filled : filled + take] = buff[:take]
            filled += take
        return out
    finally:
        sdr.deactivateStream(stream)
        sdr.closeStream(stream)
