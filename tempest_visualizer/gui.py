"""
Cross-platform graphical visualizer (Tkinter).

Runs on Windows, macOS and Linux wherever Python ships with Tk (the default on
Windows/macOS; ``sudo apt install python3-tk`` on Debian/Ubuntu). It needs
Pillow for display. No matplotlib required — the spectrum and waterfall are
rendered with numpy into images.

Panels
------
  * Source      — the image "on screen" (loaded or synthetic)
  * Spectrum    — magnitude spectrum of the simulated emanation
  * Waterfall   — per-line energy over time (shows the raster periodicity)
  * Reconstruct — the ghost frame recovered from the IQ

Controls let you change display mode, SNR, oversampling, and apply the
low-pass countermeasure to see leakage drop in real time.
"""

from __future__ import annotations

import numpy as np


def _mag_to_image(arr: np.ndarray, out_hw=None, log=False) -> "Image.Image":
    from PIL import Image

    a = np.asarray(arr, dtype=np.float64)
    if log:
        a = np.log1p(np.abs(a))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1e-6
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((a * 255).astype("uint8"), "L")
    if out_hw is not None:
        img = img.resize((out_hw[1], out_hw[0]))
    return img


def _spectrum_image(iq: np.ndarray, width=512, height=160) -> "Image.Image":
    from PIL import Image

    seg = min(len(iq), 1 << 16)
    x = np.asarray(iq[:seg])
    win = np.hanning(len(x))
    spec = np.abs(np.fft.fftshift(np.fft.fft(x * win)))
    spec = np.log1p(spec)
    spec = spec / (spec.max() + 1e-9)
    # resample to width
    idx = np.linspace(0, len(spec) - 1, width).astype(int)
    line = spec[idx]
    canvas = np.zeros((height, width), dtype=np.uint8)
    for col, v in enumerate(line):
        h = int(v * (height - 1))
        canvas[height - 1 - h :, col] = 200
    return Image.fromarray(canvas, "L")


def _waterfall_image(iq: np.ndarray, line_length: int, max_lines=200) -> "Image.Image":
    mag = np.abs(np.asarray(iq))
    n_lines = min(max_lines, mag.size // max(line_length, 1))
    if n_lines < 2:
        from PIL import Image
        return Image.new("L", (256, 128), 0)
    block = mag[: n_lines * line_length].reshape(n_lines, line_length)
    # downsample columns for display
    cols = min(512, line_length)
    idx = np.linspace(0, line_length - 1, cols).astype(int)
    return _mag_to_image(block[:, idx], log=True)


def run() -> None:  # pragma: no cover - interactive
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

    from .timing import STANDARD_TIMINGS, get_timing
    from .emanation import EmanationParams, simulate_emanation
    from .reconstruct import reconstruct_frame
    from . import countermeasures, synthetic
    from . import imageio_simple as imio

    from .livefeed import open_source, LiveProcessor
    from .emanation import EmanationParams as _EP

    state = {"image": None, "iq": None, "photos": {}, "live": None, "source": None}

    root = tk.Tk()
    root.title("TEMPEST / Deep-TEMPEST Visualizer — research & defense")
    root.geometry("1180x760")

    # ----- controls -----
    ctrl = ttk.Frame(root, padding=8)
    ctrl.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(ctrl, text="Mode:").pack(side=tk.LEFT)
    mode_var = tk.StringVar(value="1024x768@60")
    ttk.Combobox(ctrl, textvariable=mode_var, width=16,
                 values=sorted(STANDARD_TIMINGS), state="readonly").pack(side=tk.LEFT, padx=4)

    ttk.Label(ctrl, text="SNR dB:").pack(side=tk.LEFT)
    snr_var = tk.DoubleVar(value=10.0)
    ttk.Scale(ctrl, from_=-5, to=30, variable=snr_var, length=120).pack(side=tk.LEFT, padx=4)

    ttk.Label(ctrl, text="Osr:").pack(side=tk.LEFT)
    spp_var = tk.IntVar(value=1)
    ttk.Spinbox(ctrl, from_=1, to=4, textvariable=spp_var, width=3).pack(side=tk.LEFT, padx=4)

    blind_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(ctrl, text="Blind detect", variable=blind_var).pack(side=tk.LEFT, padx=6)

    cm_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(ctrl, text="Low-pass countermeasure", variable=cm_var).pack(side=tk.LEFT)
    cutoff_var = tk.DoubleVar(value=0.18)
    ttk.Scale(ctrl, from_=0.05, to=0.5, variable=cutoff_var, length=90).pack(side=tk.LEFT, padx=4)

    status = tk.StringVar(value="Load an image or use the synthetic screen, then Run.")

    # ----- image panels -----
    grid = ttk.Frame(root, padding=8)
    grid.pack(fill=tk.BOTH, expand=True)
    panels = {}
    titles = [("source", "Source (on screen)"), ("spectrum", "Emanation spectrum"),
              ("waterfall", "Per-line waterfall"), ("recon", "Reconstructed (leaked)")]
    for i, (key, title) in enumerate(titles):
        f = ttk.LabelFrame(grid, text=title, padding=4)
        f.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
        lbl = ttk.Label(f)
        lbl.pack(fill=tk.BOTH, expand=True)
        panels[key] = lbl
    grid.columnconfigure(0, weight=1)
    grid.columnconfigure(1, weight=1)
    grid.rowconfigure(0, weight=1)
    grid.rowconfigure(1, weight=1)

    def show(key, pil_img, max_w=540, max_h=300):
        from PIL import Image
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        state["photos"][key] = photo  # keep ref
        panels[key].configure(image=photo)

    def load_image():
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All", "*.*")])
        if not path:
            return
        try:
            state["image"] = imio.load_gray(path)
            show("source", _mag_to_image(state["image"]))
            status.set(f"Loaded {path}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def use_synthetic():
        t = get_timing(mode_var.get())
        state["image"] = synthetic.make_screen(t.h_active, t.v_active)
        show("source", _mag_to_image(state["image"]))
        status.set("Using synthetic screen.")

    def run_pipeline():
        if state["image"] is None:
            use_synthetic()
        timing = get_timing(mode_var.get())
        img = np.asarray(state["image"], dtype=float) / 255.0
        if cm_var.get():
            img = countermeasures.lowpass_image(img, cutoff=cutoff_var.get())
            img = np.clip(img, 0, 1)
            show("source", _mag_to_image(img))
        params = EmanationParams(samples_per_pixel=int(spp_var.get()),
                                 snr_db=float(snr_var.get()))
        iq = simulate_emanation(img, timing, params)
        state["iq"] = iq
        show("spectrum", _spectrum_image(iq))
        ll = timing.h_total * int(spp_var.get())
        show("waterfall", _waterfall_image(iq, ll))
        if blind_var.get():
            res = reconstruct_frame(iq)
        else:
            res = reconstruct_frame(iq, ll, timing.v_total)
        show("recon", _mag_to_image(res.image))
        score = countermeasures.leakage_score(img, res.image)
        status.set(
            f"mode={timing.name}  line_len={res.line_length}  frame_h={res.frame_height}"
            f"  conf={res.confidence:.2f}  leakage={score:.2f}"
            + ("  [countermeasure ON]" if cm_var.get() else "")
        )

    def save_recon():
        if state["iq"] is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if not path:
            return
        timing = get_timing(mode_var.get())
        res = reconstruct_frame(state["iq"], timing.h_total * int(spp_var.get()),
                                timing.v_total)
        imio.save_gray(path, res.image)
        status.set(f"Saved reconstruction to {path}")

    def _live_tick():
        if state["live"] is None:
            return
        source, proc = state["live"]
        try:
            frame = source.grab()
            lf = proc.process(frame)
        except Exception as exc:
            status.set(f"live stopped: {exc}")
            stop_live()
            return
        show("source", _mag_to_image(frame))
        show("spectrum", _spectrum_image(lf.iq))
        show("waterfall", _waterfall_image(lf.iq, proc.timing.h_total * proc.params.samples_per_pixel))
        show("recon", _mag_to_image(lf.recon.image))
        score = countermeasures.leakage_score(np.asarray(frame, float) / 255.0, lf.recon.image)
        status.set(f"LIVE via {source.name}  {lf.fps:4.1f} fps  "
                   f"line_len={lf.recon.line_length} frame_h={lf.recon.frame_height}  "
                   f"leakage={score:.2f}")
        state["after"] = root.after(80, _live_tick)

    def stop_live():
        if state.get("after"):
            try:
                root.after_cancel(state["after"])
            except Exception:
                pass
            state["after"] = None
        if state["live"] is not None:
            try:
                state["live"][0].close()
            except Exception:
                pass
            state["live"] = None
        live_btn.config(text="Live ▶")

    def toggle_live():
        if state["live"] is not None:
            stop_live()
            status.set("live stopped.")
            return
        timing = get_timing(mode_var.get())
        try:
            source = open_source(prefer="auto",
                                 downscale_to=(timing.v_active, timing.h_active))
        except Exception as exc:
            messagebox.showerror("Live unavailable", str(exc))
            return
        proc = LiveProcessor(timing, _EP(snr_db=float(snr_var.get())),
                             blind=blind_var.get(), samples_per_pixel=int(spp_var.get()))
        state["live"] = (source, proc)
        live_btn.config(text="Stop ■")
        _live_tick()

    for text, cmd in [("Load image…", load_image), ("Synthetic", use_synthetic),
                      ("Run ▶", run_pipeline), ("Save ghost…", save_recon)]:
        ttk.Button(ctrl, text=text, command=cmd).pack(side=tk.RIGHT, padx=3)
    live_btn = ttk.Button(ctrl, text="Live ▶", command=toggle_live)
    live_btn.pack(side=tk.RIGHT, padx=3)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_live(), root.destroy()))

    bar = ttk.Frame(root, padding=(8, 2))
    bar.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Label(bar, textvariable=status).pack(side=tk.LEFT)
    ttk.Label(bar, text="For equipment you own or are authorized to test.",
              foreground="#a33").pack(side=tk.RIGHT)

    use_synthetic()
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run()
