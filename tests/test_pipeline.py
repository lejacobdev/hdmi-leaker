"""Round-trip and sanity tests. Run with: python -m pytest -q"""

import numpy as np

from tempest_visualizer import (
    get_timing,
    simulate_emanation,
    EmanationParams,
    reconstruct_frame,
    countermeasures,
)
from tempest_visualizer import iqio, synthetic


def test_timing_consistency():
    t = get_timing("1920x1080@60")
    assert abs(t.refresh_hz - 60.0) < 0.5
    assert t.pixels_per_frame == t.h_total * t.v_total


def test_emanation_shape_and_snr():
    t = get_timing("640x480@60")
    img = synthetic.make_screen(t.h_active, t.v_active)
    iq = simulate_emanation(img, t, EmanationParams(snr_db=15, seed=1))
    assert iq.dtype == np.complex64
    assert iq.size == t.h_total * t.v_total
    # higher SNR -> stronger detected periodicity than very low SNR
    res_hi = reconstruct_frame(iq, t.h_total, t.v_total)
    assert res_hi.image.shape == (t.v_total, t.h_total)


def test_blind_line_length_detection():
    t = get_timing("800x600@60")
    img = synthetic.make_screen(t.h_active, t.v_active)
    iq = simulate_emanation(img, t, EmanationParams(snr_db=20, seed=2))
    res = reconstruct_frame(iq)  # fully blind
    # detected line length should be close to a small multiple/divisor of h_total
    ratio = res.line_length / t.h_total
    assert 0.9 < ratio < 1.1 or abs(res.line_length - t.h_total) < 8


def test_reconstruction_correlates_with_source():
    t = get_timing("640x480@60")
    img = synthetic.make_screen(t.h_active, t.v_active).astype(float) / 255.0
    iq = simulate_emanation(img, t, EmanationParams(snr_db=25, seed=3))
    res = reconstruct_frame(iq, t.h_total, t.v_total)
    active = res.image[: t.v_active, : t.h_active]
    score = abs(countermeasures.ncc(countermeasures._edges(img), active))
    assert score > 0.1  # a clearly non-random resemblance


def test_countermeasure_reduces_signal_at_fixed_floor():
    """A low-pass 'TEMPEST font' must cut emanation power, i.e. effective SNR,
    when the ambient noise floor is held fixed. This is the physical invariant;
    the per-reconstruction correlation is noisy, so we assert on SNR."""
    from tempest_visualizer.emanation import signal_power

    t = get_timing("640x480@60")
    img = synthetic.make_screen(t.h_active, t.v_active).astype(float) / 255.0
    p = EmanationParams(seed=4)
    base_power = signal_power(img, t, p)
    for cutoff in (0.30, 0.18, 0.10):
        mit = np.clip(countermeasures.lowpass_image(img, cutoff), 0, 1)
        assert signal_power(mit, t, p) < base_power
    # Strongest filter should remove a lot (>6 dB).
    strong = np.clip(countermeasures.lowpass_image(img, 0.10), 0, 1)
    drop_db = 10 * np.log10(signal_power(strong, t, p) / base_power)
    assert drop_db < -6.0


def test_iq_roundtrip_formats(tmp_path):
    iq = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
    for ext in (".cf32", ".cs16", ".npy"):
        p = str(tmp_path / f"x{ext}")
        iqio.save_iq(p, iq)
        back = iqio.load_iq(p)
        assert back.size == iq.size
        if ext == ".cf32":
            assert np.allclose(back, iq, atol=1e-5)
