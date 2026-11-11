"""
Generate synthetic 'screen' images so the toolkit is useful with zero assets.

Produces a grayscale array resembling a desktop: a title bar, some lines of
text, and high-contrast shapes — i.e. content with the sharp edges that
emanations reveal best.
"""

from __future__ import annotations

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


def make_screen(width: int = 1024, height: int = 768, text: str | None = None) -> np.ndarray:
    """Return a grayscale uint8 image of a fake screen, HxW."""
    if _HAVE_PIL:
        return _make_screen_pil(width, height, text)
    return _make_screen_numpy(width, height)


def _make_screen_pil(width: int, height: int, text: str | None) -> np.ndarray:
    img = Image.new("L", (width, height), color=25)
    d = ImageDraw.Draw(img)
    # title bar
    d.rectangle([0, 0, width, height // 12], fill=70)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", max(12, height // 28))
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, height // 18))
    except Exception:
        font = ImageFont.load_default()
        big = font
    d.text((16, height // 40), "TEMPEST demo — simulated screen content", fill=230, font=big)

    lines = (text or _DEFAULT_TEXT).strip("\n").splitlines()
    y = height // 10
    for line in lines:
        d.text((24, y), line, fill=210, font=font)
        y += max(16, height // 26)
        if y > height - 40:
            break

    # some shapes with hard edges (edges leak the most)
    d.rectangle([width - 260, height - 220, width - 40, height - 40], outline=255, width=4)
    d.ellipse([width - 240, height - 200, width - 60, height - 60], outline=200, width=3)
    for i in range(8):
        x = 40 + i * 26
        d.line([x, height - 180, x, height - 40], fill=255, width=2)

    return np.asarray(img, dtype=np.uint8)


def _make_screen_numpy(width: int, height: int) -> np.ndarray:
    arr = np.full((height, width), 25, dtype=np.uint8)
    arr[: height // 12, :] = 70
    # vertical bars (sharp edges)
    for i in range(0, width, 64):
        arr[height // 6 : height // 2, i : i + 24] = 220
    # a bright box
    arr[height // 2 : height - 40, width // 2 : width - 40] = 180
    return arr


_DEFAULT_TEXT = """
$ ssh ops@gateway.internal
Last login: Tue Nov 11 09:14:02 2025
ops@gateway:~$ cat notes.txt
  - rotate signing keys before Friday
  - TEMPEST test: shielding v2 on monitor cable
  - compare leakage score before/after
ops@gateway:~$ ./measure --cable hdmi --mode 1024x768
The quick brown fox jumps over the lazy dog. 0123456789
ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz
"""
