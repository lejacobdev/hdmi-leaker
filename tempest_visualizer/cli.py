"""
Command-line interface.

Subcommands
-----------
  modes        list known display timings
  simulate     image -> simulated emanation IQ file
  reconstruct  IQ file -> reconstructed ghost image
  demo         end-to-end: (synthetic or given) image -> IQ -> ghost, side by side
  countermeasure  compare leakage with and without a low-pass mitigation
  gui          launch the graphical visualizer
  capture      (optional) sample IQ from a SoapySDR radio you are authorized to test

Everything except ``gui`` and ``capture`` runs headless on Windows, macOS and
Linux with only numpy (+ Pillow for image files).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .timing import STANDARD_TIMINGS, get_timing
from .emanation import EmanationParams, simulate_emanation
from .reconstruct import reconstruct_frame
from . import countermeasures
from . import iqio
from . import imageio_simple as imio
from . import synthetic


def _add_emanation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default="1024x768@60", help="display timing name")
    p.add_argument("--snr", type=float, default=10.0, help="capture SNR in dB")
    p.add_argument("--spp", type=int, default=1, help="samples per pixel (oversampling)")
    p.add_argument("--freq-offset", type=float, default=0.0,
                   help="residual carrier offset (cycles/pixel)")
    p.add_argument("--phase-noise", type=float, default=0.0,
                   help="per-sample phase jitter std (radians)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")


def _params_from(args) -> EmanationParams:
    return EmanationParams(
        samples_per_pixel=args.spp,
        snr_db=args.snr,
        freq_offset_cycles_per_pixel=args.freq_offset,
        phase_noise_std=args.phase_noise,
        seed=args.seed,
    )


def cmd_modes(args) -> int:
    for name in sorted(STANDARD_TIMINGS):
        print(STANDARD_TIMINGS[name])
    return 0


def _load_image(args) -> np.ndarray:
    if getattr(args, "image", None):
        return imio.load_gray(args.image)
    t = get_timing(args.mode)
    return synthetic.make_screen(t.h_active, t.v_active)


def cmd_simulate(args) -> int:
    timing = get_timing(args.mode)
    image = _load_image(args)
    iq = simulate_emanation(image, timing, _params_from(args))
    iqio.save_iq(args.out, iq)
    print(f"wrote {iq.size} complex samples to {args.out}")
    print(f"  mode={timing.name} line_len={timing.h_total*args.spp} "
          f"frame_h={timing.v_total} snr={args.snr}dB")
    return 0


def cmd_reconstruct(args) -> int:
    iq = iqio.load_iq(args.iq)
    ll = args.line_length
    fh = args.frame_height
    if args.mode and (ll is None or fh is None):
        t = get_timing(args.mode)
        ll = ll or t.h_total * args.spp
        fh = fh or t.v_total
    res = reconstruct_frame(iq, line_length=ll, frame_height=fh)
    imio.save_gray(args.out, res.image)
    print(f"reconstructed {res.image.shape[1]}x{res.image.shape[0]} "
          f"(line_len={res.line_length}, frame_h={res.frame_height}, "
          f"confidence={res.confidence:.3f}) -> {args.out}")
    return 0


def cmd_demo(args) -> int:
    timing = get_timing(args.mode)
    image = _load_image(args)
    iq = simulate_emanation(image, timing, _params_from(args))
    # reconstruct blind (detector) unless --assist given
    if args.assist:
        res = reconstruct_frame(iq, timing.h_total * args.spp, timing.v_total)
    else:
        res = reconstruct_frame(iq)
    score = countermeasures.leakage_score(
        np.asarray(image, float) / 255.0, res.image
    )
    _save_sidebyside(args.out, image, res.image,
                     labels=("original (on screen)", "reconstructed (leaked)"))
    print(f"demo -> {args.out}")
    print(f"  detected line_len={res.line_length} frame_h={res.frame_height} "
          f"confidence={res.confidence:.3f}")
    print(f"  leakage score (0..1, higher = more legible leak): {score:.3f}")
    return 0


def cmd_countermeasure(args) -> int:
    from .emanation import signal_power, sigma_for_snr

    timing = get_timing(args.mode)
    image = _load_image(args).astype(float) / 255.0

    # Fix the noise floor to the *unmitigated* image's SNR, then hold it constant
    # so the mitigation competes against a real, fixed ambient noise level.
    base_params = _params_from(args)
    floor = sigma_for_snr(signal_power(image, timing, base_params), args.snr)

    noise_power = 2.0 * floor * floor

    def eff_snr_db(img):
        return 10.0 * np.log10(signal_power(img, timing, base_params) / noise_power)

    def run(img, seed):
        params = _params_from(args)
        params.noise_sigma = floor
        params.seed = seed
        iq = simulate_emanation(img, timing, params)
        res = reconstruct_frame(iq, timing.h_total * args.spp, timing.v_total)
        return res.image, countermeasures.leakage_score(image, res.image)

    mit = np.clip(countermeasures.lowpass_image(image, cutoff=args.cutoff), 0, 1)

    # Effective SNR is deterministic and physical; the correlation score is
    # averaged over several noise realisations to tame its variance.
    base_snr, mit_snr = eff_snr_db(image), eff_snr_db(mit)
    base_recon, _ = run(image, 0)
    mit_recon, _ = run(mit, 0)
    base_score = float(np.mean([run(image, s)[1] for s in range(6)]))
    mit_score = float(np.mean([run(mit, s)[1] for s in range(6)]))

    _save_sidebyside(
        args.out, base_recon, mit_recon,
        labels=(f"no mitigation  SNR={base_snr:.1f}dB leak={base_score:.2f}",
                f"low-pass {args.cutoff}  SNR={mit_snr:.1f}dB leak={mit_score:.2f}"),
    )
    print(f"countermeasure -> {args.out}")
    print(f"  effective receiver SNR:  {base_snr:5.1f} dB  ->  {mit_snr:5.1f} dB "
          f"({mit_snr - base_snr:+.1f} dB, less is harder to eavesdrop)")
    print(f"  reconstruction leakage:  {base_score:.3f}    ->  {mit_score:.3f} "
          f"(avg of 6 runs; 0=unreadable, 1=clear)")
    return 0


def cmd_gui(args) -> int:
    try:
        from .gui import run as run_gui
        run_gui()
    except ModuleNotFoundError as exc:
        print(f"GUI unavailable: {exc}.\n"
              "Tkinter/Pillow may be missing. On Debian/Ubuntu: "
              "'sudo apt install python3-tk'. Windows/macOS Python include Tk.\n"
              "Meanwhile the headless CLI (modes/simulate/reconstruct/demo/"
              "countermeasure) works without a display.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"GUI error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_capture(args) -> int:
    from .capture import capture_block
    iq = capture_block(
        sample_rate=args.sample_rate,
        center_freq=args.freq,
        num_samples=args.num_samples,
        gain_db=args.gain,
        device_args=args.device,
        confirm_authorized=args.i_am_authorized,
    )
    iqio.save_iq(args.out, iq)
    print(f"captured {iq.size} samples -> {args.out}")
    return 0


def _save_sidebyside(path, left, right, labels=("", "")):
    """Stack two images horizontally and save (needs Pillow; falls back to npy)."""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        np.save(path + ".left.npy", np.asarray(left))
        np.save(path + ".right.npy", np.asarray(right))
        return

    def to_img(a):
        a = np.asarray(a, dtype=float)
        a = a * 255.0 if a.max() <= 1.0 else a
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"), "L")

    li, ri = to_img(left), to_img(right)
    h = max(li.height, ri.height)
    w = li.width + ri.width + 20
    canvas = Image.new("L", (w, h + 24), color=0)
    canvas.paste(li, (0, 24))
    canvas.paste(ri, (li.width + 20, 24))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), labels[0], fill=255)
    d.text((li.width + 24, 6), labels[1], fill=255)
    canvas.save(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tempest-visualizer",
        description="Educational TEMPEST / Deep-TEMPEST emanation simulator, "
                    "reconstructor, and countermeasure evaluator.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("modes", help="list known display timings")
    sp.set_defaults(func=cmd_modes)

    sp = sub.add_parser("simulate", help="image -> simulated emanation IQ")
    _add_emanation_args(sp)
    sp.add_argument("--image", help="input image (default: synthetic screen)")
    sp.add_argument("--out", default="emanation.cf32", help="output IQ file")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("reconstruct", help="IQ -> ghost image")
    sp.add_argument("iq", help="input IQ file")
    sp.add_argument("--out", default="reconstructed.png")
    sp.add_argument("--mode", default=None, help="assist detector with a known mode")
    sp.add_argument("--spp", type=int, default=1)
    sp.add_argument("--line-length", type=int, default=None)
    sp.add_argument("--frame-height", type=int, default=None)
    sp.set_defaults(func=cmd_reconstruct)

    sp = sub.add_parser("demo", help="end-to-end image -> IQ -> ghost")
    _add_emanation_args(sp)
    sp.add_argument("--image", help="input image (default: synthetic screen)")
    sp.add_argument("--assist", action="store_true",
                    help="give the reconstructor the known mode instead of blind detect")
    sp.add_argument("--out", default="demo.png")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("countermeasure", help="measure a low-pass mitigation")
    _add_emanation_args(sp)
    sp.add_argument("--image", help="input image (default: synthetic screen)")
    sp.add_argument("--cutoff", type=float, default=0.18,
                    help="low-pass cutoff (0..1); lower = stronger mitigation")
    sp.add_argument("--out", default="countermeasure.png")
    sp.set_defaults(func=cmd_countermeasure)

    sp = sub.add_parser("gui", help="launch the graphical visualizer")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser(
        "capture",
        help="(optional) capture IQ from a SoapySDR radio you are authorized to test",
    )
    sp.add_argument("--sample-rate", type=float, required=True)
    sp.add_argument("--freq", type=float, required=True, help="center frequency Hz")
    sp.add_argument("--num-samples", type=int, required=True)
    sp.add_argument("--gain", type=float, default=None)
    sp.add_argument("--device", default="", help="SoapySDR device args")
    sp.add_argument("--out", default="capture.cf32")
    sp.add_argument("--i-am-authorized", action="store_true",
                    help="acknowledge you own or are authorized to test the source")
    sp.set_defaults(func=cmd_capture)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
