"""
TEMPEST / Deep-TEMPEST educational visualizer.

A cross-platform, research- and education-oriented toolkit for understanding
how the raster signal driving an HDMI/DVI display unintentionally radiates
("compromising emanations", a.k.a. TEMPEST / Van Eck phreaking), and how such
leakage can be modeled, reconstructed, and — importantly — *defended against*.

Scope and intent
----------------
This package is built for:

  * learning the signal theory behind video-cable electromagnetic leakage,
  * self-assessment of equipment you own or are explicitly authorized to test,
  * demonstrating and measuring the effectiveness of countermeasures
    (shielding, TEMPEST-safe fonts, resolution/refresh choices).

It deliberately ships a *forward model* (simulate leakage from an image you
provide) and a *reconstruction* stage that operates on IQ files you captured
from your own equipment. It is not a turnkey interception tool and contains no
targeting, tuning-for-a-victim, or detection-evasion functionality.

Intercepting emanations from equipment you do not own or are not authorized to
test is illegal in most jurisdictions. Use responsibly.
"""

from .timing import DisplayTiming, STANDARD_TIMINGS, get_timing
from .emanation import simulate_emanation, EmanationParams
from .reconstruct import reconstruct_frame, detect_line_length, ReconstructResult
from . import countermeasures

__version__ = "0.1.0"

__all__ = [
    "DisplayTiming",
    "STANDARD_TIMINGS",
    "get_timing",
    "simulate_emanation",
    "EmanationParams",
    "reconstruct_frame",
    "detect_line_length",
    "ReconstructResult",
    "countermeasures",
    "__version__",
]
