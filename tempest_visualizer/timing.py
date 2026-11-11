"""
Standard display timings (CEA-861 / VESA DMT).

A video link does not only transmit the visible pixels: every frame also
carries horizontal and vertical *blanking* intervals (the legacy of CRT beam
retrace). Blanking is why the leaked signal has a very specific horizontal and
vertical periodicity — and that periodicity is exactly what a reconstruction
stage locks onto. Getting these numbers right is the whole game.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayTiming:
    """Pixel-accurate timing for one video mode.

    Attributes
    ----------
    name:
        Human readable label, e.g. ``"1920x1080@60"``.
    h_active, v_active:
        Visible resolution in pixels / lines.
    h_total, v_total:
        Total pixels per line / lines per frame *including* blanking.
    pixel_clock_hz:
        Pixel clock in Hz. ``pixel_clock = h_total * v_total * refresh``.
    """

    name: str
    h_active: int
    v_active: int
    h_total: int
    v_total: int
    pixel_clock_hz: float

    @property
    def refresh_hz(self) -> float:
        return self.pixel_clock_hz / (self.h_total * self.v_total)

    @property
    def line_rate_hz(self) -> float:
        """Horizontal (line) frequency in Hz."""
        return self.pixel_clock_hz / self.h_total

    @property
    def pixels_per_frame(self) -> int:
        return self.h_total * self.v_total

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name}: {self.h_active}x{self.v_active} active, "
            f"{self.h_total}x{self.v_total} total, "
            f"{self.pixel_clock_hz/1e6:.3f} MHz pixel clock, "
            f"{self.refresh_hz:.2f} Hz refresh"
        )


# A compact table of common modes. Values follow VESA DMT / CEA-861.
STANDARD_TIMINGS: dict[str, DisplayTiming] = {
    t.name: t
    for t in [
        DisplayTiming("640x480@60", 640, 480, 800, 525, 25_175_000),
        DisplayTiming("800x600@60", 800, 600, 1056, 628, 40_000_000),
        DisplayTiming("1024x768@60", 1024, 768, 1344, 806, 65_000_000),
        DisplayTiming("1280x720@60", 1280, 720, 1650, 750, 74_250_000),
        DisplayTiming("1280x1024@60", 1280, 1024, 1688, 1066, 108_000_000),
        DisplayTiming("1366x768@60", 1366, 768, 1792, 798, 85_500_000),
        DisplayTiming("1600x900@60", 1600, 900, 1800, 1000, 108_000_000),
        DisplayTiming("1920x1080@60", 1920, 1080, 2200, 1125, 148_500_000),
        DisplayTiming("1920x1200@60RB", 1920, 1200, 2080, 1235, 154_000_000),
        DisplayTiming("2560x1440@60", 2560, 1440, 2720, 1481, 241_500_000),
    ]
}


def get_timing(name: str) -> DisplayTiming:
    """Look up a timing by name, with a helpful error listing valid names."""
    try:
        return STANDARD_TIMINGS[name]
    except KeyError:
        raise KeyError(
            f"Unknown timing {name!r}. Known modes: "
            + ", ".join(sorted(STANDARD_TIMINGS))
        ) from None
