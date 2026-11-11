"""
Live feed: run the leakage simulation on real, moving screen content.

What "live" means here
----------------------
The frames that become your monitor's HDMI output are your machine's
*framebuffer*. This module grabs that framebuffer (an ordinary screen capture of
**your own** display), pushes each frame through the same forward model +
reconstruction used elsewhere, and yields the result continuously — so you watch
the "leaked" ghost update as your screen changes.

This is screen mirroring into the simulator. It is **not** reading a cable or
radio. It only ever sees the display of the machine it runs on. For capturing
the actual electromagnetic emanations of equipment *you own or are authorized to
test*, use ``capture.py`` (SDR), which is gated behind an authorization flag.

Frame sources, in order of preference:
  * ``mss``            — fast, cross-platform (Windows/macOS/Linux-X11). Optional.
  * PIL ``ImageGrab``  — Windows/macOS fallback (and Linux with scrot).
  * synthetic animated — always available, for demos with no display/capture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .timing import DisplayTiming
from .emanation import EmanationParams, simulate_emanation
from .reconstruct import reconstruct_frame, ReconstructResult


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class FrameSource:
    """Yields grayscale uint8 frames (HxW)."""

    name = "base"

    def grab(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional
        pass


class MSSSource(FrameSource):
    name = "mss"

    def __init__(self, monitor: int = 1, downscale_to: tuple[int, int] | None = None):
        import mss  # type: ignore

        self._sct = mss.mss()
        mons = self._sct.monitors
        idx = monitor if 0 <= monitor < len(mons) else 1
        self._mon = mons[idx]
        self._down = downscale_to

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self._mon)
        arr = np.asarray(shot)[..., :3]  # BGRA -> BGR
        gray = arr[..., ::-1] @ np.array([0.299, 0.587, 0.114])  # to RGB weights
        gray = gray.astype(np.uint8)
        if self._down is not None:
            gray = _resize_nn(gray, *self._down)
        return gray

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


class ImageGrabSource(FrameSource):
    name = "imagegrab"

    def __init__(self, downscale_to: tuple[int, int] | None = None):
        from PIL import ImageGrab  # type: ignore

        self._grab = ImageGrab.grab
        self._down = downscale_to
        # Probe once so construction fails fast where it is unsupported.
        self._grab()

    def grab(self) -> np.ndarray:
        img = self._grab().convert("L")
        gray = np.asarray(img, dtype=np.uint8)
        if self._down is not None:
            gray = _resize_nn(gray, *self._down)
        return gray


class SyntheticSource(FrameSource):
    """Animated fake desktop so 'live' works with no display or capture backend."""

    name = "synthetic"

    def __init__(self, width: int = 1024, height: int = 768):
        self._w, self._h = width, height
        self._t0 = time.time()

    def grab(self) -> np.ndarray:
        from . import synthetic

        elapsed = int((time.time() - self._t0))
        blink = "_" if elapsed % 2 == 0 else " "
        scroll = "." * (elapsed % 40)
        text = (
            "$ ssh ops@gateway.internal\n"
            f"ops@gateway:~$ tail -f /var/log/app.log {scroll}\n"
            f"[{time.strftime('%H:%M:%S')}] frame {elapsed:04d} processing{blink}\n"
            "  - rotate signing keys before Friday\n"
            "  - TEMPEST test: shielding v2 on monitor cable\n"
            "The quick brown fox jumps over the lazy dog. 0123456789\n"
        )
        return synthetic.make_screen(self._w, self._h, text=text)


def open_source(
    prefer: str = "auto",
    downscale_to: tuple[int, int] | None = None,
    monitor: int = 1,
) -> FrameSource:
    """Open the best available frame source.

    ``prefer`` in {"auto", "mss", "imagegrab", "synthetic"}. ``auto`` tries real
    screen capture first and falls back to the synthetic animated source so the
    live mode always runs (e.g. on a headless box).
    """
    order = {
        "auto": ["mss", "imagegrab", "synthetic"],
        "mss": ["mss"],
        "imagegrab": ["imagegrab"],
        "synthetic": ["synthetic"],
    }.get(prefer, ["synthetic"])

    last_err: Exception | None = None
    for kind in order:
        try:
            if kind == "mss":
                return MSSSource(monitor=monitor, downscale_to=downscale_to)
            if kind == "imagegrab":
                return ImageGrabSource(downscale_to=downscale_to)
            if kind == "synthetic":
                w, h = (downscale_to[1], downscale_to[0]) if downscale_to else (1024, 768)
                return SyntheticSource(width=w, height=h)
        except Exception as exc:  # backend missing / no display
            last_err = exc
            continue
    raise RuntimeError(f"No usable frame source ({last_err})")


# --------------------------------------------------------------------------- #
# Live processing
# --------------------------------------------------------------------------- #
@dataclass
class LiveFrame:
    source: np.ndarray          # grayscale uint8, the captured screen
    iq: np.ndarray              # simulated emanation IQ for this frame
    recon: ReconstructResult    # reconstructed ghost
    fps: float


class LiveProcessor:
    """Turns captured frames into simulated-and-reconstructed live frames."""

    def __init__(
        self,
        timing: DisplayTiming,
        params: EmanationParams | None = None,
        blind: bool = False,
        samples_per_pixel: int = 1,
    ):
        self.timing = timing
        self.params = params or EmanationParams()
        self.params.samples_per_pixel = samples_per_pixel
        self.blind = blind
        self._last = time.time()

    def process(self, frame: np.ndarray) -> LiveFrame:
        iq = simulate_emanation(frame, self.timing, self.params)
        if self.blind:
            recon = reconstruct_frame(iq)
        else:
            ll = self.timing.h_total * self.params.samples_per_pixel
            recon = reconstruct_frame(iq, ll, self.timing.v_total)
        now = time.time()
        fps = 1.0 / max(now - self._last, 1e-6)
        self._last = now
        return LiveFrame(source=frame, iq=iq, recon=recon, fps=fps)


def _resize_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape[:2]
    ys = np.clip((np.arange(out_h) * in_h / out_h).astype(np.int64), 0, in_h - 1)
    xs = np.clip((np.arange(out_w) * in_w / out_w).astype(np.int64), 0, in_w - 1)
    return arr[ys][:, xs]
