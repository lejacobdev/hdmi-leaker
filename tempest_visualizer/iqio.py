"""
IQ file I/O in the de-facto SDR interchange formats.

Supported:
  * ``.cf32`` / ``.fc32`` / ``.complex64`` — interleaved 32-bit float I,Q
    (GNU Radio / SoapySDR "CF32"). This is the default.
  * ``.cs16`` / ``.sc16`` — interleaved 16-bit signed I,Q (many SDRs' native
    format), scaled to +/-1.0 on read.
  * ``.cs8`` / ``.sc8`` — interleaved 8-bit signed I,Q (e.g. RTL-SDR style).

These are raw headerless formats, so the format is taken from the file
extension. ``.npy`` is also accepted for convenience.
"""

from __future__ import annotations

import os

import numpy as np

_FLOAT_EXTS = {".cf32", ".fc32", ".complex64", ".cfile"}
_S16_EXTS = {".cs16", ".sc16"}
_S8_EXTS = {".cs8", ".sc8"}


def save_iq(path: str, iq: np.ndarray) -> None:
    """Write a complex64 IQ array to ``path``; format chosen by extension."""
    iq = np.asarray(iq).astype(np.complex64)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, iq)
        return
    if ext in _S16_EXTS:
        inter = _interleave(iq, 32767.0).astype(np.int16)
    elif ext in _S8_EXTS:
        inter = _interleave(iq, 127.0).astype(np.int8)
    else:  # default: CF32
        inter = _interleave(iq, 1.0).astype(np.float32)
    inter.tofile(path)


def load_iq(path: str) -> np.ndarray:
    """Read an IQ file into a complex64 array; format chosen by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.complex64)
    if ext in _S16_EXTS:
        raw = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    elif ext in _S8_EXTS:
        raw = np.fromfile(path, dtype=np.int8).astype(np.float32) / 128.0
    elif ext in _FLOAT_EXTS:
        raw = np.fromfile(path, dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown IQ extension {ext!r}. Use one of: "
            ".cf32/.fc32/.complex64/.cfile, .cs16/.sc16, .cs8/.sc8, .npy"
        )
    if raw.size % 2 != 0:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def _interleave(iq: np.ndarray, scale: float) -> np.ndarray:
    out = np.empty(iq.size * 2, dtype=np.float32)
    out[0::2] = iq.real * scale
    out[1::2] = iq.imag * scale
    return out
